"""Base image classifier class that support clustering correction loss"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, Sequence, TypeAlias

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from safetensors import torch as st
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification import multiclass_accuracy

from ..correction import (
    class_otm_matching,
    clustering_loss,
    louvain_communities,
)
from ..datasets.huggingface import HuggingFaceDataset
from ..utils import get_reasonable_n_jobs, load_tensor_batched, make_tqdm

Batch: TypeAlias = dict[str, Tensor]
ClusteringMethod: TypeAlias = Literal["louvain", "dbscan", "hdbscan"]

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


class BaseClassifier(pl.LightningModule):
    """
    See module documentation

    Warning:
        When subclassing this, remember that the forward method must be able to
        deal with either `Tensor` or `Batch` inputs, and must return a logit
        `Tensor`.
    """

    n_classes: int

    lcc_submodules: list[str]
    clst_weight: float
    lcc_kwargs: dict[str, Any]
    ce_weight: float

    image_key: Any
    label_key: Any
    logit_key: Any

    # Used during training with correction
    _y_true: Tensor
    _n_classes: int
    _clustering: dict[str, tuple[np.ndarray, dict[int, set[int]]]]

    # pylint: disable=unused-argument
    def __init__(
        self,
        n_classes: int,
        lcc_submodules: list[str] | None = None,
        clst_weight: float = 1,
        lcc_kwargs: dict[str, Any] | None = None,
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
            clst_weight (float, optional): Weight of the clustering loss in the
                clustering-CE loss. Ignored if `lcc_submodules` is `None` or
                `[]`
            lcc_kwargs (dict, optional): Passed to the correction loss function.
                Ignored if `lcc_submodules` is `None` or `[]`
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
        self.save_hyperparameters(ignore=["model"])
        self.n_classes = n_classes
        self.lcc_submodules = (
            []
            if lcc_submodules is None
            else [
                (sm if sm.startswith("model.") else "model." + sm)
                for sm in lcc_submodules
            ]
        )
        self.clst_weight, self.lcc_kwargs = clst_weight, lcc_kwargs or {}
        self.ce_weight = ce_weight
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
        compute_cl = (
            stage == "train" and self.lcc_submodules and self.clst_weight > 0
        )
        if compute_cl:
            idx, lst = batch["_idx"].cpu().numpy(), []
            for sm, z in latent.items():
                y_clst = self._clustering[sm][0][idx]
                match = self._clustering[sm][1]
                p = y_clst >= 0  # Exclude outliers
                u = clustering_loss(
                    z=z[p],
                    y_true=y[p],
                    y_clst=y_clst[p],
                    matching=match,
                    n_true_classes=self._n_classes,
                )
                lst.append(u)
            loss_clst = torch.stack(lst).mean()
            loss = self.ce_weight * loss_ce + self.clst_weight * loss_clst
        else:
            loss_clst, loss = torch.tensor(0.0), loss_ce
        if stage:
            log = {
                f"{stage}/loss": loss,
                f"{stage}/ce": loss_ce,
                f"{stage}/acc": multiclass_accuracy(
                    logits, y, num_classes=self.n_classes, average="micro"
                ),
            }
            if compute_cl:
                log[f"{stage}/cl"] = loss_clst
            self.log_dict(log, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
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
            return {"optimizer": optimizer, "scheduler": scheduler}
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
                if not isinstance(out, Tensor):
                    raise ValueError(
                        f"Unsupported latent object type: {type(out)}."
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

    def on_train_batch_end(self, *args, **kwargs) -> None:
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

    def on_train_start(self) -> None:
        """
        Stores the entire dataset label vector in `_y_true` and the number
        of classes in `_n_classes`
        """
        if self.lcc_submodules and self.clst_weight > 0:
            dm = self.trainer.datamodule  # type: ignore
            assert isinstance(dm, HuggingFaceDataset)
            self._y_true = dm.y_true("train")
            self._n_classes = dm.n_classes("train")
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        """
        Performs dataset-wide clustering and stores the results in
        `_clustering`.

        Warning:
            Because full dataset latent clustering must be run on the CPU, the
            whole model must be moved to the CPU for this step. This makes
            training on multiple GPU a bit difficult. At best, the model can be
            ran on a single GPU and switch back and forth between CPU and GPU.
            But if the model is ran on multiple GPUs, then FDSL will end up
            being done multiple times, once per process.
        """
        if self.lcc_submodules and self.clst_weight > 0:
            joblib_config = {
                "backend": "loky",
                "n_jobs": get_reasonable_n_jobs(),
                "verbose": 0,
            }
            with (
                joblib.parallel_backend(**joblib_config),
                TemporaryDirectory() as tmp,
            ):
                self._clustering = full_dataset_latent_clustering(
                    model=self,
                    dataset=self.trainer.datamodule,  # type: ignore
                    output_dir=tmp,
                    method=self.hparams["clustering_method"],
                    device="cuda",
                    scaling="standard",
                    tqdm_style="console",
                )
            nor = [  # Non-outlier ratio
                (y_clst >= 0).sum() / len(y_clst)
                for _, (y_clst, _) in self._clustering.items()
            ]
            self.log("train/nor", np.mean(nor), on_epoch=True)
            # ↓ fake clustering
            # self._clustering = {
            #     sm: (
            #         np.zeros_like(self.trainer.datamodule.y_true("train")),
            #         {i: {i} for i in range(self._n_classes)},
            #     )
            #     for sm in self.lcc_submodules
            # }
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        """Cleans up training epoch specific temporary attributes."""
        if hasattr(self, "_clustering"):
            del self._clustering
        return super().on_train_epoch_end()

    # pylint: disable=arguments-differ
    def test_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "train")

    def validation_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "val")


def full_dataset_latent_clustering(
    model: BaseClassifier,
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    method: ClusteringMethod = "louvain",
    device: Literal["cpu", "cuda"] | None = None,
    scaling: Literal["standard", "minmax"] | None = "standard",
    classes: list[int] | None = None,
    split: Literal["train", "val", "test"] = "train",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[str, tuple[np.ndarray, dict[int, set[int]]]]:
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
        k (int, optional):
        classes (list[int] | None, optional): If specified, then only the
            specified classes are considered for clustering (however, all
            samples are still evaluated regardless of class). Use this if there
            are too many classes, or if the dataset is just too large to fit in
            memory (e.g.  ImageNet).

    Returns:
        dict[str, tuple[np.ndarray, dict[int, set[int]]]]: A dictionary that
            maps a submodule name to a tuple containing 1. the cluster labels,
            and 2. the matching dict as returned by
            `nlnas.correction.class_otm_matching`. The submodule in question
            are those specified in `model.lcc_submodules`.
    """
    dl = dataset.get_dataloader(split)
    output_dir, tqdm = Path(output_dir) / split, make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(tqdm(dl, "Evaluating", leave=False)):
            todo = [
                sm
                for sm in model.lcc_submodules
                if not (output_dir / f"{sm}.{idx:04}.st").exists()
            ]
            if not todo:
                continue
            out: dict[str, Tensor] = {}
            model.forward_intermediate(
                inputs=batch[model.image_key],
                submodules=todo,
                output_dict=out,
                keep_gradients=False,
            )
            for sm, z in out.items():
                st.save_file({"": z}, output_dir / f"{sm}.{idx:04}.st")
        model.train()
    clst: dict[str, tuple[np.ndarray, dict[int, set[int]]]] = {}
    y_true = dataset.y_true(split)
    mask = (
        None if classes is None else torch.isin(y_true, torch.tensor(classes))
    )
    y_true = y_true[mask] if mask is not None else y_true
    for sm in tqdm(model.lcc_submodules, "Clustering", leave=False):
        z = load_tensor_batched(
            output_dir, sm, mask=mask, tqdm_style="console"
        )
        y_clst = get_cluster_labels(z, method, scaling, device)
        matching = class_otm_matching(y_true[: z.shape[0]], y_clst)
        clst[sm] = (y_clst, matching)
    return clst


# pylint: disable=duplicate-code
def get_cluster_labels(
    z: np.ndarray | Tensor,
    method: ClusteringMethod = "louvain",
    scaling: Literal["standard", "minmax"] | None = "standard",
    device: Literal["cpu", "cuda"] | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Self-explanatory

    Args:
        z (np.ndarray | Tensor):
        method (Literal["louvain", "dbscan", "hdbscan"], optional):
        scaling (Literal["standard", "minmax"] | None, optional):
        device (Literal["cpu", "cuda"] | None, optional):
        **kwargs: Passed to the clustering object
    """
    use_cuda = (
        device == "cuda" or device is None
    ) and torch.cuda.is_available()

    if use_cuda:
        from cuml.cluster import DBSCAN, HDBSCAN
        from cuml.preprocessing import MinMaxScaler, StandardScaler
    else:
        from sklearn.cluster import DBSCAN, HDBSCAN
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if isinstance(z, Tensor):
        z = z.cpu().detach().numpy()
    z = z.reshape(len(z), -1)
    if scaling == "standard":
        z = StandardScaler().fit_transform(z)
    elif scaling == "minmax":
        z = MinMaxScaler().fit_transform(z)

    if method == "louvain":
        return louvain_communities(z, device=device, **kwargs)[1]
    if method == "dbscan":
        return DBSCAN(**kwargs).fit(z).labels_
    if method == "hdbscan":
        return HDBSCAN(**kwargs).fit(z).labels_
    raise ValueError(f"Unsupported clustering method '{method}'")
