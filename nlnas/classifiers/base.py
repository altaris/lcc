"""Base image classifier class that support clustering correction loss"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, Sequence, TypeAlias

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
from safetensors import numpy as st
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification import multiclass_accuracy

from nlnas.correction.clustering import lcc_loss, otm_matching_predicates
from nlnas.datasets.emetd_dataloader import EMETDDataLoader

from ..correction import (
    LCCClassSelection,
    choose_classes,
    class_otm_matching,
    lcc_targets,
    louvain_communities,
)
from ..datasets import (
    BatchedTensorDataset,
    HuggingFaceDataset,
    load_tensor_batched,
)
from ..utils import (
    check_cuda,
    get_reasonable_n_jobs,
    make_tqdm,
    to_array,
)
from .utils import (
    OPTIMIZERS,
    SCHEDULERS,
    log_optimizers_lr,
    validate_lcc_kwargs,
)

Batch: TypeAlias = dict[str, Tensor]


@dataclass
class LatentClusteringData:
    """
    Convenience struct that holds some latent clustering correction data for a
    given latent space.
    """

    p: np.ndarray
    """
    `(N,)` boolean vector that marks misclustered samples (regardless of
    true class).
    """

    matching: dict[int, set[int]]
    """
    Matching between the true and cluster classes. See
    `nlnas.correction.clustering.class_otm_matching`.
    """

    targets: dict[int, Tensor]
    """
    Dict that maps a true class to a correctly clustered sample in that true
    class. Note that not every true class may be represented in this dict.
    """

    y_clst: np.ndarray
    """`(N,)` vector of cluster labels."""


class BaseClassifier(pl.LightningModule):
    """
    Base image classifier class that supports LCC.

    Warning:
        When subclassing this, remember that the forward method must be able to
        deal with either `Tensor` or `Batch` inputs, and must return a logit
        `Tensor`.
    """

    _lcc_data: dict[str, LatentClusteringData] | None = None
    """
    If LCC is applied, then this is non `None` and updated at the begining of
    each epoch. See also `full_dataset_latent_clustering`.
    """

    def __init__(
        self,
        n_classes: int,
        lcc_submodules: list[str] | None = None,
        lcc_kwargs: dict | None = None,
        ce_weight: float = 1,
        image_key: Any = 0,
        label_key: Any = 1,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_classes (int):
            lcc_submodules (list[str] | None, optional): Submodules to consider
                for the latent correction loss. If `None` or `[]`, LCC is not
                performed
            lcc_kwargs (dict | None, optional): Optional parameters for LCC.
                Expected entries (all optional) are:
                * **weight (float):** Defaults to $10^{-4}$
                * **class_selection
                    (`nlnas.correction.LCCClassSelection` | None):** Defaults to
                    `None`, which means all classes are considered for
                    correction
                * **interval (int):** Apply LCC every `interval` epochs.
                    Defaults to $1$, meaning LCC will be applied every epoch
                    (after warmup).
                * **warmup (int):** Number of epochs to wait before
                    starting LCC. Defaults to $0$, meaning LCC will start
                    immediately.
                * **k (int):** Number of nearest neighbors to consider for LCC,
                  and Louvain clustering.
                * **pca_dim (int):** Samples are reduced to this dimension
                  before constructing the KNN graph. This must be at most the
                  batch size.
            ce_weight (float, optional): Weight of the cross-entropy loss in the
                clustering-CE loss. Ignored if LCC is not applied. Defaults to
                $1$.
            image_key (Any, optional): A batch passed to the model can be a
                tuple (most common) or a dict. This parameter specifies the key
                to use to retrieve the input tensor.
            label_key (Any, optional): Analogous to `image_key`.
            optimizer (str, optional): Optimizer name, case insensitive.
                See `nlnas.classifier.base.OPTIMIZERS` and
                https://pytorch.org/docs/stable/optim.html#algorithms .
            optimizer_kwargs (dict | None, optional): Forwarded to the optimizer
                constructor
            scheduler (str | None, optional): Scheduler name, case insensitive. See
                `nlnas.classifier.base.SCHEDULERS`. If left to `None`, then no
                scheduler is used.
            scheduler_kwargs (dict | None, optional): Forwarded to the
                scheduler, if any.
            kwargs: Forwarded to
                [`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#)
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "ce_weight",
            "image_key",
            "label_key",
            "lcc_kwargs",
            "lcc_submodules",
            "n_classes",
            "optimizer_kwargs",
            "optimizer",
            "scheduler_kwargs",
            "scheduler",
        )
        if lcc_submodules:
            validate_lcc_kwargs(lcc_kwargs)

    def _evaluate(self, batch: Batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        image_key = self.hparams["image_key"]
        label_key = self.hparams["label_key"]
        x, y = batch[image_key], batch[label_key].to(self.device)
        latent: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.lcc_submodules, latent, keep_gradients=True
        )
        assert isinstance(logits, Tensor)
        loss_ce = nn.functional.cross_entropy(logits, y.long())
        if self._lcc_data and stage == "train":
            idx = to_array(batch["_idx"])
            _losses = [
                lcc_loss(
                    z=z,
                    y_true=y,
                    y_clst=self._lcc_data[sm].y_clst[idx],
                    matching=self._lcc_data[sm].matching,
                    targets=self._lcc_data[sm].targets,
                    n_true_classes=self.hparams["n_classes"],
                )
                for sm, z in latent.items()
            ]
            loss_lcc = (
                torch.stack(_losses).mean()
                if _losses
                else torch.tensor(0.0, requires_grad=True)
                # ↑ actually need grad?
            )
            lcc_weight = self.hparams.get("lcc_kwargs", {}).get("weight", 1e-4)
            loss = self.hparams["ce_weight"] * loss_ce + lcc_weight * loss_lcc
            self.log(f"{stage}/lcc", loss_lcc, sync_dist=True)
        else:
            loss = loss_ce
        if stage:
            self.log_dict(
                {f"{stage}/loss": loss, f"{stage}/ce": loss_ce},
                sync_dist=True,
            )
            self.log(
                f"{stage}/acc",
                multiclass_accuracy(
                    logits,
                    y,
                    num_classes=self.hparams["n_classes"],
                    average="micro",
                ),
                prog_bar=True,
                sync_dist=True,
            )
        return loss  # type: ignore

    def configure_optimizers(self) -> Any:
        cls = OPTIMIZERS[self.hparams["optimizer"].lower()]
        optimizer = cls(
            self.parameters(),
            **(self.hparams.get("optimizer_kwargs") or {}),
        )
        if self.hparams["scheduler"]:
            cls = SCHEDULERS[self.hparams["scheduler"]]
            scheduler = cls(
                optimizer,
                **(self.hparams.get("scheduler_kwargs") or {}),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                },
            }
        return optimizer

    def forward_intermediate(
        self,
        inputs: Tensor | Batch | list[Tensor] | Sequence[Batch],
        submodules: list[str],
        output_dict: dict,
        keep_gradients: bool = False,
    ) -> Tensor | list[Tensor]:
        """
        Runs the model and collects the output of specified submodules. The
        intermediate outputs are stored in `output_dict` under the
        corresponding submodule name. In particular, this method has side
        effects.

        Args:
            x (Tensor | Batch | list[Tensor] | list[Batch]): If batched (i.e.
                `x` is a list), then so is the output of this function and the
                entries in the `output_dict`
            submodules (list[str]):
            output_dict (dict):
            keep_gradients (bool, optional): If `True`, the tensors in
                `output_dict` keep their gradients (if they had some on the
                first place). If `False`, they are detached and moved to the
                CPU.
        """

        def maybe_detach(x: Tensor) -> Tensor:
            return x if keep_gradients else x.detach().cpu()

        def create_hook(key: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(_model: nn.Module, _args: Any, out: Any) -> None:
                if (
                    isinstance(out, (list, tuple))
                    and len(out) == 1
                    and isinstance(out[0], Tensor)
                ):
                    out = out[0]
                elif (  # Special case for ViTs
                    isinstance(out, (list, tuple))
                    and len(out) == 2
                    and isinstance(out[0], Tensor)
                    and not isinstance(out[1], Tensor)
                ):
                    out = out[0]
                elif not isinstance(out, Tensor):
                    raise ValueError(
                        f"Unsupported latent object type: {type(out)}: {out}."
                    )
                if batched:
                    if key not in output_dict:
                        output_dict[key] = []
                    output_dict[key].append(maybe_detach(out))
                else:
                    output_dict[key] = maybe_detach(out)

            return hook

        batched = isinstance(inputs, (list, tuple))
        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        if batched:
            logits = [
                maybe_detach(
                    self.forward(
                        batch
                        if isinstance(batch, Tensor)
                        else batch[self.hparams["image_key"]]
                    )
                )
                for batch in inputs
            ]
        else:
            logits = maybe_detach(  # type: ignore
                self.forward(
                    inputs
                    if isinstance(inputs, Tensor)
                    else inputs[self.hparams["image_key"]]
                )
            )
        for h in handles:
            h.remove()
        return logits

    @staticmethod
    def get_image_processor(model_name: str, **kwargs: Any) -> Callable:
        """
        Returns an image processor for the model. By defaults, returns the
        identity function.
        """
        return lambda input: input

    @property
    def lcc_submodules(self) -> list[str]:
        """
        Returns the list of submodules considered for LCC, whith correct prefix
        if needed.

        TODO:
            Move to :class:`nlnas.classifier.WrappedClassifier` instead.
        """
        return (
            []
            if not self.hparams["lcc_submodules"]
            else [
                (sm if sm.startswith("model.") else "model." + sm)
                for sm in self.hparams["lcc_submodules"]
            ]
        )

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """Just logs all optimizer's learning rate"""
        log_optimizers_lr(self, sync_dist=True)
        super().on_train_batch_end(*args, **kwargs)

    def on_train_epoch_end(self) -> None:
        """Cleans up training specific temporary attributes"""
        self._lcc_data = None
        super().on_train_epoch_end()

    def on_train_epoch_start(self) -> None:
        """
        Performs dataset-wide latent clustering and stores the results in
        private attribute `BaseClassifier._lc_data`.
        """
        # wether to apply LCC this epoch
        lcc_kwargs = self.hparams.get("lcc_kwargs") or {}
        do_lcc = (
            # we are passed warmup (lcc_warmup being None is equivalent to no
            # warmup)...
            self.current_epoch >= (lcc_kwargs.get("warmup") or 0)
            and (
                # ... and an LCC interval is specified...
                lcc_kwargs.get("interval") is not None
                # ... and the current epoch can have LCC done...
                and self.current_epoch % int(lcc_kwargs.get("interval", 1))
                == 0
            )
            # ... and there are submodule selected for LCC...
            and self.lcc_submodules
            # ... and the LCC weight is non-zero
            and lcc_kwargs.get("weight", 0) > 0
        )
        if do_lcc:
            if self.trainer.global_rank == 0:
                joblib_config = {
                    "backend": "loky",
                    "n_jobs": get_reasonable_n_jobs(),
                    "verbose": 0,
                }
                with (
                    joblib.parallel_backend(**joblib_config),
                    TemporaryDirectory() as tmp,
                ):
                    self._lcc_data = full_dataset_latent_clustering(
                        model=self,
                        dataset=self.trainer.datamodule,  # type: ignore
                        output_dir=tmp,
                        device="cuda",
                        # classes=lcc_kwargs.get("class_selection"),
                        tqdm_style="console",
                    )
            else:
                self._lcc_data = None
            if self.trainer.world_size >= 2:
                if self._lcc_data is not None:
                    for d in self._lcc_data.values():
                        for k, v in d.targets.items():
                            d.targets[k] = v.cpu()
                self._lcc_data = self.trainer.strategy.broadcast(
                    self._lcc_data, src=0
                )
                assert isinstance(self._lcc_data, dict)
                for d in self._lcc_data.values():
                    for k, v in d.targets.items():
                        d.targets[k] = v.to(self.device)
        super().on_train_epoch_start()

    def on_train_start(self) -> None:
        """
        Explicitly registers hyperparameters and metrics. You'd think Lightning
        would do this automatically, but nope.
        """
        self.logger.log_hyperparams(  # type: ignore
            self.hparams,
            {
                s + "/" + m: np.nan
                for s, m in product(
                    ["train", "val"], ["acc", "loss", "ce", "lcc"]
                )
            },
        )
        super().on_train_start()

    def test_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.test_step`."""
        return self._evaluate(batch, "test")

    def training_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.training_step`."""
        return self._evaluate(batch, "train")

    def validation_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.validation_step`."""
        return self._evaluate(batch, "val")


def _fde_pca(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    pca_dim: dict[str, int | None],
    split: Literal["train", "val", "test"] = "train",
    device: Any = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    `full_dataset_evaluation` in the case where PCA dim-redux has to be applied.
    See `full_dataset_evaluation` for the precise meaning of the arguments. Note
    that in this case, `pca_dim` mist be a dict that maps LCC submodule names to
    PCA dimensions.
    """
    use_cuda, device = check_cuda(device)
    if use_cuda:
        from cuml import IncrementalPCA

        pcas = {
            sm: IncrementalPCA(n_components=d, output_type="numpy")
            for sm, d in pca_dim.items()
            if d
        }
    else:
        from sklearn.decomposition import IncrementalPCA

        pcas = {
            sm: IncrementalPCA(n_components=d)
            for sm, d in pca_dim.items()
            if d
        }

    dl = dataset.get_dataloader(split)
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        if use_cuda:
            model.to("cuda")
        for idx, batch in enumerate(tqdm(dl, "Fitting PCA(s)", leave=False)):
            out: dict[str, Tensor] = {}
            model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=model.lcc_submodules,
                output_dict=out,
                keep_gradients=False,
            )
            for sm, z in out.items():
                if sm in pcas:
                    pcas[sm].partial_fit(z.flatten(1))
        for idx, batch in enumerate(tqdm(dl, "Evaluating", leave=False)):
            out = {}
            y_pred = model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=model.lcc_submodules,
                output_dict=out,
                keep_gradients=False,
            )
            assert isinstance(y_pred, Tensor)
            out["y_pred"] = y_pred
            for sm, z in out.items():
                if sm in pcas:
                    z = pcas[sm].transform(z.flatten(1))
                st.save_file(
                    {"": to_array(z)},
                    output_dir / f"{sm}.{idx:04}.st",
                )
        model.train()


def _fde_no_pca(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    split: Literal["train", "val", "test"] = "train",
    device: Any = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    `full_dataset_evaluation` in the case where no PCA dim-redux is to be
    applied. See `full_dataset_evaluation` for the precise meaning of the
    arguments.
    """
    use_cuda, device = check_cuda(device)
    dl = dataset.get_dataloader(split)
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        if use_cuda:
            model.to("cuda")
        for idx, batch in enumerate(tqdm(dl, "Evaluating", leave=False)):
            todo = [
                sm
                for sm in model.lcc_submodules
                if not (output_dir / f"{sm}.{idx:04}.st").exists()
            ]
            if not todo:
                continue
            out: dict[str, Tensor] = {}
            y_pred = model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=todo,
                output_dict=out,
                keep_gradients=False,
            )
            assert isinstance(y_pred, Tensor)
            out["y_pred"] = y_pred
            for sm, z in out.items():
                st.save_file(
                    {"": to_array(z.flatten(1))},
                    output_dir / f"{sm}.{idx:04}.st",
                )
        model.train()


def _y_true_mask(
    y_true: Tensor,
    classes: list[int] | LCCClassSelection | None = None,
    y_true_batch_dir: str | Path | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> Tensor:
    # ↓ classes is just a list of classes
    if isinstance(classes, list):
        return torch.isin(y_true, torch.tensor(classes))
    # ↓ classes is a LCCClassSelection policy (e.g. "max_connected")
    if isinstance(classes, str):
        if y_true_batch_dir is None:
            raise ValueError(
                "The batch directory of the true label vector is required for "
                f"LCC policy '{classes}'."
            )
        y_pred = load_tensor_batched(
            y_true_batch_dir, "y_pred", tqdm_style=tqdm_style
        )
        assert isinstance(y_pred, Tensor)  # For typechecking
        classes = choose_classes(y_true, y_pred, policy=classes)
        if classes:
            return torch.isin(y_true, torch.tensor(classes))
        return torch.full_like(y_true, True, dtype=torch.bool)
    # ↓ Leaving classes to None means all classes are considered, so no mask
    return torch.full_like(y_true, True, dtype=torch.bool)


def full_dataset_evaluation(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    pca_dim: int | dict[str, int | None] | None = None,
    split: Literal["train", "val", "test"] = "train",
    device: Any = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Evaluate model on whole dataset and saves latent representations and
    prediction batches in `output_dir`. The naming pattern is

        <output_dir> / <submodule>.<batch_idx>.st

    and `batch_idx` is a zero-padded 4-digit number. The files are
    [Safetensors](https://huggingface.co/docs/safetensors/index) files.

    Args:
        model (BaseClassifier):
        dataset (HuggingFaceDataset):
        output_dir (str | Path):
        pca_dim (int | dict[str, int | None] | None, optional): If specified, a
            PCA is fitted and applied to the latent representations. This is
            useful when the latent dimensions are very large. However, it
            becomes necessary to iterate over the dataset twice, which more than
            doubles the execution time of this method. It is possible to specify
            a PCA dimension for each or some of the latent spaces.
        split (Literal['train', 'val', 'test'], optional): Defaults to "train".
        device (Any, optional): Defaults to `None`, which automatically finds
            the best device.
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):
            Defaults to `None`, mearning no progress bar.
    """
    if pca_dim is None:
        _fde_no_pca(
            model,
            dataset,
            output_dir,
            split=split,
            device=device,
            tqdm_style=tqdm_style,
        )
    else:
        if isinstance(pca_dim, int):
            pca_dim = {sm: pca_dim for sm in model.lcc_submodules}
        _fde_pca(
            model,
            dataset,
            output_dir,
            pca_dim,
            split=split,
            device=device,
            tqdm_style=tqdm_style,
        )


# TODO: Re-enable support for non-trivial class selection policies
def full_dataset_latent_clustering(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    device: Any = None,
    # classes: list[int] | LCCClassSelection | None = None,
    split: Literal["train", "val", "test"] = "train",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[str, LatentClusteringData]:
    """
    Performs latent clustering and matching (against true labels) on the full
    dataset in one go. Since holding all latent representation tensors in memory
    isn't realistic, some (aka. a shitload of) temporary tensor files are
    created in `<output_dir>/<split>`. See
    `nlnas.classifiers.full_dataset_evaluation`.

    Warning:
        The temporary tensor files created by this method are not deleted. You
        need to clean them up manually. Or don't if you want to keep them.

    Warning:
        Don't forget to execute `dataset.setup("fit")` before calling this
        method =)

    Args:
        model (BaseClassifier):
        dataset (HuggingFaceDataset):
        output_dir (str | Path):
        classes (list[int] | LCCClassSelection | None, optional): If specified,
            then only the specified true classes are considered for clustering
            (however, all samples are still evaluated regardless of class). Use
            this if there are too many true classes, or if the dataset is just
            too large to fit in memory (e.g. ImageNet). See also
            `nlnas.correction.LCC_CLASS_SELECTIONS`.

    Returns:
        A dictionary that maps a submodule name to its latent clustering data,
        see `nlnas.classifiers.LatentClusteringData`.
    """

    output_dir = Path(output_dir) / split
    _, device = check_cuda(device)
    full_dataset_evaluation(
        model,
        dataset,
        output_dir=output_dir,
        split=split,
        device=device,
        tqdm_style=tqdm_style,
    )
    y_true = dataset.y_true(split)
    # mask = _y_true_mask(y_true, classes, output_dir, tqdm_style)
    # if not mask.all():
    #     y_true = y_true[mask]

    result: dict[str, LatentClusteringData] = {}
    tqdm = make_tqdm(tqdm_style)
    lcc_kwargs = model.hparams.get("lcc_kwargs", {})
    for sm in tqdm(model.lcc_submodules, "Clustering", leave=False):
        ds = BatchedTensorDataset(batch_dir=output_dir, prefix=sm, key="")
        dl = EMETDDataLoader(
            ds,
            # mask=mask,
            device=device,
            batch_size=256,
            shuffle=False,
        )
        _, y_clst = louvain_communities(
            dl,
            k=lcc_kwargs.get("k", 5),
            device=device,
            tqdm_style=tqdm_style,
        )
        matching = class_otm_matching(y_true, y_clst)
        _, _, p, _ = otm_matching_predicates(y_true, y_clst, matching)
        p = p.sum(axis=0) > 0  # Select MC samples regardless of true class
        targets = lcc_targets(
            dl, y_true, y_clst, matching, n_true_classes=dataset.n_classes()
        )
        # for i_true, (p, x) in targets.items():
        #     targets[i_true] = (_inflate_vector(p, mask), x)
        # y_clst=_inflate_vector(y_clst, mask)
        # p_mc=_inflate_vector(p_mc, mask)
        result[sm] = LatentClusteringData(
            matching=matching,
            p=p,
            targets=targets,
            y_clst=y_clst,
        )

    return result
