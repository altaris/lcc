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
from safetensors import numpy as st
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification import multiclass_accuracy

from ..correction import (
    ClusteringMethod,
    LCCClassSelection,
    choose_classes,
    class_otm_matching,
    get_cluster_labels,
    lcc_knn_indices,
    lcc_loss,
    lcc_targets,
)
from ..datasets import HuggingFaceDataset
from ..logging import r0_debug
from ..utils import (
    get_reasonable_n_jobs,
    load_tensor_batched,
    make_tqdm,
    to_array,
)

Batch: TypeAlias = dict[str, Tensor]


OPTIMIZERS: dict[str, type] = {
    "asgd": torch.optim.ASGD,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adamax": torch.optim.Adamax,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "optimizer": torch.optim.Optimizer,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
    "sparseadam": torch.optim.SparseAdam,
}

SCHEDULERS: dict[str, type] = {
    "constantlr": torch.optim.lr_scheduler.ConstantLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "cycliclr": torch.optim.lr_scheduler.CyclicLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "lambdalr": torch.optim.lr_scheduler.LambdaLR,
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "multiplicativelr": torch.optim.lr_scheduler.MultiplicativeLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "polynomiallr": torch.optim.lr_scheduler.PolynomialLR,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "sequentiallr": torch.optim.lr_scheduler.SequentialLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
}


@dataclass
class LatentClusteringData:
    """
    Convenience struct that holds some latent clustering correction data for a
    given latent space. Contents are self-explanatory.
    """

    y_clst: np.ndarray  # (n_samples,)
    matching: dict[int, set[int]]
    knn_indices: dict[int, tuple[Any, Tensor]]


# pylint: disable=arguments-differ
class BaseClassifier(pl.LightningModule):
    """
    See module documentation

    Warning:
        When subclassing this, remember that the forward method must be able to
        deal with either `Tensor` or `Batch` inputs, and must return a logit
        `Tensor`.
    """

    n_classes: int  # TODO: use hparams instead
    image_key: Any  # TODO: use hparams instead
    label_key: Any  # TODO: use hparams instead
    logit_key: Any  # TODO: use hparams instead

    # Used during training with LCC
    _lc_data: dict[str, LatentClusteringData] | None = None

    # pylint: disable=unused-argument
    def __init__(
        self,
        n_classes: int,
        lcc_submodules: list[str] | None = None,
        lcc_weight: float | None = None,
        lcc_kwargs: dict[str, Any] | None = None,
        lcc_class_selection: LCCClassSelection | None = None,
        lcc_interval: int | None = None,
        lcc_warmup: int | None = None,
        ce_weight: float = 1,
        image_key: Any = 0,
        label_key: Any = 1,
        logit_key: Any = None,
        optimizer: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        clustering_method: ClusteringMethod = "louvain",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_classes (int):
            lcc_submodules (list[str] | None, optional): Submodules to consider
                for the latent correction loss. If `None` or `[]`, LCC is not
                performed
            lcc_weight (float, optional): Weight of the clustering loss in the
                clustering-CE loss. Ignored if `lcc_submodules` is `None` or
                `[]`
            lcc_kwargs (dict, optional): Passed to the correction loss function.
                Ignored if `lcc_submodules` is `None` or `[]`
            lcc_class_selection (LCCClassSelection, optional): How to select
                (true) classes whose samples will undergo LCC. If the dataset is
                large, it might not be desirable to perform LCC on the whole
                dataset. See `nlnas.correction.choice.LCCClassSelection` for
                more information. Defaults to `"all"` which means full-dataset
                LCC.
            lcc_interval (int, optional): Apply LCC every `lcc_interval` epochs.
                If set to `None`, LCC is not performed.
            lcc_warmup (int, optional): Number of epochs to wait before starting
                LCC. Setting this to `None` is the same as setting it to 0.
            ce_weight (float, optional): Weight of the cross-entropy loss in the
                clustering-CE loss. Ignored if `lcc_submodules` is `None` or
                `[]`
            image_key (Any, optional): A batch passed to the model can be a
                tuple (most common) or a dict. This parameter specifies the key
                to use to retrieve the input tensor.
            label_key (Any, optional): Analogous to `image_key`
            logit_key (Any, optional): Analogous to `image_key` and `label_key`
                but used to extract the logits from the model's output. Leave
                to `None` if the model already returns logit tensors. If
                `model`is a Hugging Face transformer that outputs a
                [`ImageClassifierOutput`](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput)
                or a
                [`ImageClassifierOutputWithNoAttention`](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput),
                then this key should be
                `"logits"`.
            optimizer (str, optional): Optimizer name, case insensitive. See
                `OPTIMIZERS` and
                https://pytorch.org/docs/stable/optim.html#algorithms .
            optimizer_kwargs (dict, optional): Forwarded to the optimizer
                constructor
            scheduler (str, optional): Scheduler name, case insensitive. See
                `SCHEDULERS`. If left to `None`, then no scheduler is used.
            scheduler_kwargs (dict, optional): Forwarded to the scheduler, if
                any.
            clustering_method (ClusteringMethod, optional): See
                `full_dataset_latent_clustering`. Only relevant if
                `lcc_submodules` is specified (i.e. the model is to undergo
                latent clustering correction).
            kwargs: Forwarded to
                [`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#)
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "ce_weight",
            "clustering_method",
            "lcc_class_selection",
            "lcc_interval",
            "lcc_kwargs",
            "lcc_submodules",
            "lcc_warmup",
            "lcc_weight",
            "optimizer_kwargs",
            "optimizer",
            "scheduler_kwargs",
            "scheduler",
        )
        self.n_classes = n_classes
        self.image_key, self.label_key = image_key, label_key
        self.logit_key = logit_key

    def _evaluate(self, batch: Batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        x, y = batch[self.image_key], batch[self.label_key].to(self.device)
        latent: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.lcc_submodules, latent, keep_gradients=True
        )
        assert isinstance(logits, Tensor)
        loss_ce = nn.functional.cross_entropy(logits, y.long())
        if self._lc_data:
            idx, _losses = batch["_idx"].cpu(), []
            for sm, z in latent.items():
                targets = lcc_targets(
                    z,
                    y_true=y,
                    y_clst=self._lc_data[sm].y_clst[idx],
                    matching=self._lc_data[sm].matching,
                    knn_indices=self._lc_data[sm].knn_indices,
                    n_true_classes=self.n_classes,
                )
                _losses.append(lcc_loss(z, targets))
            loss_lcc = torch.stack(_losses).mean()
            loss = (
                self.hparams["ce_weight"] * loss_ce
                + self.hparams["lcc_weight"] * loss_lcc
            )
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
                    logits, y, num_classes=self.n_classes, average="micro"
                ),
                prog_bar=True,
                sync_dist=True,
            )
        return loss

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

        def create_hook(key: str):
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
            logits = [  # type: ignore
                maybe_detach(
                    self.forward(
                        batch
                        if isinstance(batch, Tensor)
                        else batch[self.image_key]
                    )
                )
                for batch in inputs
            ]
        else:
            logits = maybe_detach(  # type: ignore
                self.forward(
                    inputs
                    if isinstance(inputs, Tensor)
                    else inputs[self.image_key]
                )
            )
        for h in handles:
            h.remove()
        return logits

    @staticmethod
    def get_image_processor(model_name: str, **kwargs) -> Callable:
        """
        Returns an image processor for the model. By defaults, returns the
        identity function.
        """
        return lambda input: input

    @property
    def lcc_submodules(self) -> list[str]:
        return (
            []
            if self.hparams["lcc_submodules"] is None
            else [
                (sm if sm.startswith("model.") else "model." + sm)
                for sm in self.hparams["lcc_submodules"]
            ]
        )

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """Just logs all optimizer's learning rate"""

        def _lr(o: torch.optim.Optimizer) -> float:
            return o.param_groups[0]["lr"]

        opts = self.optimizers()
        if isinstance(opts, list):
            self.log_dict(
                {
                    f"lr_{i}": _lr(opt)
                    for i, opt in enumerate(opts)
                    if isinstance(opt, torch.optim.Optimizer)
                },
                sync_dist=True,
            )
        elif isinstance(opts, torch.optim.Optimizer):
            self.log("lr", _lr(opts), sync_dist=True)
        return super().on_train_batch_end(*args, **kwargs)

    def on_train_epoch_end(self) -> None:
        """Cleans up training specific temporary attributes"""
        self._lc_data = None
        return super().on_train_end()

    def on_train_epoch_start(self) -> None:
        """
        Performs dataset-wide latent clustering and stores the results in
        `_lc_data`.
        """
        # wether to apply LCC this epoch
        do_lcc = (
            # we are passed warmup (lcc_warmup being None is equivalent to no
            # warmup)...
            self.current_epoch >= (self.hparams["lcc_warmup"] or 0)
            and (
                # ... and an LCC interval is specified...
                self.hparams["lcc_interval"] is not None
                # ... and the current epoch can have LCC done...
                and self.current_epoch % int(self.hparams["lcc_interval"]) == 0
            )
            # ... and there are submodule selected for LCC...
            and self.lcc_submodules
            # ... and the LCC weight is non-zero
            and (self.hparams["lcc_weight"] or 0) > 0
        )
        if do_lcc:
            joblib_config = {
                "backend": "loky",
                "n_jobs": get_reasonable_n_jobs(),
                "verbose": 0,
            }
            with (
                joblib.parallel_backend(**joblib_config),
                TemporaryDirectory() as tmp,
            ):
                self._lc_data = full_dataset_latent_clustering(
                    model=self,
                    dataset=self.trainer.datamodule,  # type: ignore
                    output_dir=tmp,
                    method=self.hparams["clustering_method"],
                    max_dim=None,  # no dim-redux
                    device="cuda",
                    scaling="standard",
                    classes=self.hparams["lcc_class_selection"],
                    tqdm_style="console",
                )
            log = {}
            for sm, d in self._lc_data.items():
                outlier_ratio = (d.y_clst < 0).sum() / d.y_clst.shape[0]
                log["train/outl_r/" + sm] = outlier_ratio
                log["train/n_clusters/" + sm] = len(np.unique(d.y_clst))
            self.log_dict(log, sync_dist=True)
        return super().on_train_epoch_start()

    def on_train_start(self) -> None:
        """
        Explicitly registers hyperparameters and metrics. You'd think Lightning
        would do this automatically, but nope.
        """
        self.logger.log_hyperparams(  # type: ignore
            self.hparams,  # type: ignore
            {
                s + "/" + m: np.nan
                for s, m in product(
                    ["train", "val"], ["acc", "loss", "ce", "lcc"]
                )
            },
        )
        return super().on_train_start()

    def test_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "test")

    def training_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "train")

    def validation_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "val")


def _inflate_vector(
    v: np.ndarray | Tensor | list[float],
    mask: np.ndarray | Tensor | list[bool],
) -> np.ndarray:
    """
    Say `v` has shape (n_a,) while `mask` has shape (n_b,). This function
    "inflates" `v` into a vector `w` of shape (n_b,) such that `v = w[mask]`.
    Values of `w` that don't fall in the mask are set to -1.
    """
    v, mask = to_array(v), to_array(mask).astype(bool)
    w = np.full_like(mask, -1, dtype=v.dtype)
    w[mask] = v
    return w


def full_dataset_evaluation(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    split: Literal["train", "val", "test"] = "train",
    max_dim: int | None = 8192,
    device: Literal["cpu", "cuda"] | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Evaluate model on whole dataset and saves latent representations and
    prediction batches in `output_dir`.
    """
    use_cuda = (
        device == "cuda" or device is None
    ) and torch.cuda.is_available()
    if use_cuda:
        from cuml import PCA
    else:
        from sklearn.decomposition import PCA

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
                inputs=batch[model.image_key],
                submodules=todo,
                output_dict=out,
                keep_gradients=False,
            )
            assert isinstance(y_pred, Tensor)
            out["y_pred"] = y_pred
            for sm, z in out.items():
                z = z.flatten(1)
                if max_dim is not None and z.shape[-1] > max_dim:
                    # Sadly, cuml.PCA requires a numpy array, meaning that a
                    # copy of z will be loaded back on the GPU. So the batch
                    # size or max_dim should be small enough so that the GPU can
                    # hold 2 batches at the same time plus the PCA results.
                    z = PCA(n_components=max_dim).fit_transform(to_array(z))
                st.save_file(
                    {"": to_array(z)}, output_dir / f"{sm}.{idx:04}.st"
                )
        model.train()


def full_dataset_latent_clustering(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    method: ClusteringMethod = "louvain",
    max_dim: int | None = 8192,
    device: Literal["cpu", "cuda"] | None = None,
    scaling: Literal["standard", "minmax"] | None = "standard",
    classes: list[int] | LCCClassSelection | None = None,
    split: Literal["train", "val", "test"] = "train",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[str, LatentClusteringData]:
    """
    Performs latent clustering and matching (against true labels) on the full
    dataset in one go. Since holding all latent representation tensors in
    memory isn't realistic, some (aka. a shitload of) temporary tensor files
    are created in `<output_dir>/train`.

    Warning:
        The temporary tensor files created by this method are not deleted. You
        need to clean them up manually.

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
            too large to fit in memory (e.g. ImageNet).
        max_dim (int, optional): If the dimension of a latent space is larger
            than this, then the latent representation undergo batch-wise PCA to
            make them `max_dim`-dimensional. Defaults to $8192 = 2^{13}$. Can be
            set to `None` to never resort to PCA.

    Returns:
        A dictionary that maps a submodule name to its latent clustering data,
        see `LatentClusteringData`.
    """

    output_dir = Path(output_dir) / split
    full_dataset_evaluation(
        model,
        dataset,
        output_dir=output_dir,
        split=split,
        max_dim=max_dim,
        device=device,
        tqdm_style=tqdm_style,
    )
    y_true = dataset.y_true(split)

    # ↓ classes is just a list of classes
    if isinstance(classes, list):
        mask = torch.isin(y_true, torch.tensor(classes))
        y_true = y_true[mask]

    # ↓ classes is a LCCClassSelection policy (e.g. "max_connected")
    elif isinstance(classes, str):
        y_pred = load_tensor_batched(
            output_dir, "y_pred", tqdm_style=tqdm_style
        )
        assert isinstance(y_pred, Tensor)  # For typechecking
        classes = choose_classes(y_true, y_pred, policy=classes)
        if classes:
            mask = torch.isin(y_true, torch.tensor(classes))
            y_true = y_true[mask]
        else:
            mask = torch.full_like(y_true, True, dtype=torch.bool)

    # ↓ Leaving classes to None means all classes are considered, so no mask
    else:
        mask = torch.full_like(y_true, True, dtype=torch.bool)

    result: dict[str, LatentClusteringData] = {}
    tqdm = make_tqdm(tqdm_style)
    for sm in tqdm(model.lcc_submodules, "Clustering", leave=False):
        z = load_tensor_batched(
            output_dir, sm, mask=mask, tqdm_style=tqdm_style
        )
        y_clst = get_cluster_labels(z, method, scaling, device)
        matching = class_otm_matching(y_true, y_clst)
        indices = lcc_knn_indices(z, y_true, y_clst, matching, device=device)
        result[sm] = LatentClusteringData(
            y_clst=_inflate_vector(y_clst, mask),
            matching=matching,
            knn_indices=indices,
        )

    return result
