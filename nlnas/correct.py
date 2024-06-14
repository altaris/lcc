"""Latent clustering correction"""

# pylint: disable=duplicate-code

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import turbo_broccoli as tb
from loguru import logger as logging
from safetensors import torch as st
from torch import Tensor, nn
from tqdm import tqdm

from nlnas.utils import load_tensor_batched

from .classifiers import HuggingFaceClassifier
from .correction import (
    class_otm_matching,
    clustering_loss,
    louvain_communities,
)
from .datasets import HuggingFaceDataset
from .finetune import make_trainer
from .logging import r0_info
from .training import checkpoint_ves


def _fit(
    model: HuggingFaceClassifier,
    dataset: HuggingFaceDataset,
    max_epochs: int = 100,
    k: int = 128,
):
    """Main correction training loop"""
    dataset.setup("fit")
    image_key, label_key = model.image_key, model.label_key
    optimizer = model.configure_optimizers()
    for _ in tqdm(range(max_epochs), "Correction"):
        with TemporaryDirectory() as path:
            logging.debug(
                "Setting temporary evaluation directory to '{}'", path
            )
            clustering = full_dataset_latent_clustering(
                model, dataset, Path(path), k
            )
            idx = 0
            progress = tqdm(
                dataset.train_dataloader(), "Updating weights", leave=False
            )
            for batch in progress:
                x, y_true = batch[image_key], batch[label_key]
                out: dict[str, Tensor] = {}
                logits = model.forward_intermediate(
                    inputs=batch[image_key],
                    submodules=model.cor_submodules,
                    output_dict=out,
                    keep_gradients=True,
                )
                assert isinstance(logits, Tensor)  # for typechecking
                loss_ce = nn.functional.cross_entropy(logits, y_true)
                loss_cl = torch.stack(
                    [
                        clustering_loss(
                            z=z,
                            y_true=y_true,
                            y_clst=clustering[sm][0][idx : idx + len(x)],
                            matching=clustering[sm][1],
                            k=k,
                            n_true_classes=dataset.n_classes(),
                            device="cpu",  # TODO: pass everything to CUDA
                        )
                        for sm, z in out.items()
                    ]
                )
                loss = loss_ce + model.cor_weight * loss_cl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix(
                    {
                        "corr/loss": float(loss.round(decimals=2)),
                        "corr/ce": float(loss_ce.round(decimals=2)),
                        "corr/cl": float(loss_cl.round(decimals=2)),
                    }
                )
                idx += len(x)


def correct(
    model_name: str,
    ckpt_path: Path | None,
    dataset_name: str,
    output_dir: Path,
    correction_submodules: list[str],
    correction_weight: float = 1e-3,
    max_epochs: int = 100,
    batch_size: int = 64,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    image_key: str = "image",
    label_key: str = "label",
    logit_key: str = "logits",
    head_name: str | None = None,
):
    """
    Performs latent clustering correction on a pretrained model.

    Args:
        model_name (str):
        ckpt_path (Path | None): If `None`, the correction will start from the
            weights available on the Hugging Face model hub.
        dataset_name (str):
        output_dir (Path):
        correction_submodules (list[str]):
        correction_weight (float, optional):
        max_epochs (int, optional):
        batch_size (int, optional):
        train_split (str, optional):
        val_split (str, optional):
        test_split (str, optional):
        image_key (str, optional):
        label_key (str, optional):
        logit_key (str, optional):
        head_name (str | None, optional):
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    _dataset_name = dataset_name.replace("/", "-")
    _model_name = model_name.replace("/", "-")
    _output_dir = output_dir / _dataset_name / _model_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        fit_split=train_split,
        val_split=val_split,
        test_split=test_split,
        label_key=label_key,
        image_processor=model_name,
    )
    n_classes = dataset.n_classes()

    model = HuggingFaceClassifier(
        model_name=model_name,
        n_classes=n_classes,
        head_name=head_name,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        optimizer="adam",
        optimizer_kwargs={
            "lr": 5e-5,
            "weight_decay": 0,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        # scheduler="linearlr",
        cor_weight=correction_weight,
        cor_submodules=correction_submodules,
    )
    if isinstance(ckpt_path, Path):
        # pylint: disable=no-value-for-parameter
        model.model = HuggingFaceClassifier.load_from_checkpoint(
            ckpt_path
        ).model
        r0_info("Loaded checkpoint '{}'", ckpt_path)

    trainer = make_trainer(
        _model_name, _output_dir, max_epochs=max_epochs, accelerator="cpu"
    )
    start = datetime.now()
    _fit(model, dataset, max_epochs=max_epochs)
    fit_time = datetime.now() - start
    r0_info("Finished correction in {}", fit_time)

    ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
    ckpt = ckpt.relative_to(output_dir)
    v, e, s = checkpoint_ves(ckpt)
    r0_info("Best checkpoint path: '{}'", ckpt)
    r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)

    # test_results = trainer.test(model, dataset)

    document = {
        "model": {"name": model_name},
        "dataset": {
            "name": dataset_name,
            "n_classes": n_classes,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "image_key": image_key,
            "label_key": label_key,
            "batch_size": batch_size,
        },
        "correction": {
            "hparams": dict(model.hparams),
            "epochs": max_epochs,
            "correction_submodules": correction_submodules,
            "correction_weight": correction_weight,
            "best_checkpoint": {
                "path": str(ckpt),
                "version": v,
                "best_epoch": e,
                "n_steps": s,
            },
            "time": fit_time / timedelta(seconds=1),
            # "test": test_results,
        },
    }
    tb.save_json(document, _output_dir / "results.json")


def full_dataset_latent_clustering(
    model: HuggingFaceClassifier,
    dataset: HuggingFaceDataset,
    output_dir: Path,
    k: int = 128,
) -> dict[str, tuple[np.ndarray, dict[int, set[int]]]]:
    """
    Performs latent clustering and matching (against true labels) on the full
    dataset in one go. Since holding all latent representation tensors in
    memory isn't realistic, some (aka. a shitload of) temporary tensor files
    are created in `output_dir`.

    Warning:
        The temporary tensor files created by this method are not deleted. You
        need to clean them up manually.

    Warning:
        Don't forget to execute `dataset.setup("fit")` before calling this
        method =)

    Args:
        model (HuggingFaceClassifier):
        dataset (HuggingFaceDataset):
        output_dir (Path):
        k (int, optional):

    Returns:
        dict[str, tuple[np.ndarray, dict[int, set[int]]]]: A dictionary that
            maps a submodule name to a tuple containing 1. the cluster labels,
            and 2. the matching dict as returned by
            `nlnas.correction.class_otm_matching`. The submodule in question
            are those specified in `model.cor_submodules`.
    """
    for idx, batch in enumerate(
        tqdm(dataset.train_dataloader(), "Evaluating", leave=False)
    ):
        out: dict[str, Tensor] = {}
        model.forward_intermediate(
            inputs=batch[model.image_key],
            submodules=model.cor_submodules,
            output_dict=out,
            keep_gradients=False,
        )
        for sm, z in out.items():
            st.save_file({"": z}, output_dir / f"{sm}.{idx:04}.st")
    clustering: dict[str, tuple[np.ndarray, dict[int, set[int]]]] = {}
    for sm in tqdm(model.cor_submodules, "Clustering", leave=False):
        z = load_tensor_batched(output_dir, sm, tqdm_style="console")
        _, y_clst = louvain_communities(z, k=k)
        matching = class_otm_matching(dataset.y_true("train"), y_clst)
        clustering[sm] = (y_clst, matching)
    return clustering
