"""CLI module"""

import os
from pathlib import Path

import click
from loguru import logger as logging


@click.group()
@click.option(  # --logging-level
    "--logging-level",
    default=os.getenv("LOGGING_LEVEL", "info"),
    help=(
        "Logging level, case insensitive. Defaults to 'info'. Can also be set "
        "using the LOGGING_LEVEL environment variable."
    ),
    type=click.Choice(
        ["critical", "debug", "error", "info", "warning"],
        case_sensitive=False,
    ),
)
@logging.catch
def main(logging_level: str):
    """nlnas CLI"""
    from .logging import setup_logging

    setup_logging(logging_level)


@main.command()
@click.argument("model_name", type=str)
@click.argument("dataset_name", type=str)
@click.argument(  # output_dir
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),  # type: ignore
)
@click.option(  # --lcc-submodules
    "-ls",
    "--lcc-submodules",
    default="",
    help="Comma-separated list of submodule where to apply LCC.",
    type=str,
)
@click.option(  # --lcc-weight
    "-lw",
    "--lcc-weight",
    default=1,
    help="Weight of the clustering loss term. Defaults to 1.",
    type=float,
)
@click.option(  # --lcc-interval
    "-li",
    "--lcc-interval",
    default=1,
    help="Apply LCC every n epochs. Defaults to 1 (every epoch).",
    type=int,
)
@click.option(  # --lcc-warmup
    "-lm",
    "--lcc-warmup",
    default=0,
    help="Nb. of epoch to wait for LCC. Defaults to 0 (no warmup).",
    type=int,
)
@click.option(  # --lcc-k
    "-k",
    "--lcc-k",
    default=5,
    help="Number of neigh. to consider for LCC. Defaults to 5 neigh.",
    type=int,
)
@click.option(  # --ce-weight
    "-cw",
    "--ce-weight",
    default=1,
    help=(
        "Weight of the CE loss term. Defaults to 1e-3. Ignored if LCC isn't "
        "performed."
    ),
    type=float,
)
@click.option(  # --ckpt-path
    "-c",
    "--ckpt-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=Path
    ),  # type: ignore
    default=None,
    help=(
        "Path to the checkpoint to start the correction from. If left "
        "unspecified, the correction will start from the weights available on "
        "the Hugging Face model hub."
    ),
)
@click.option(  # --max-epochs
    "-e",
    "--max-epochs",
    default=100,
    help=(
        "Maximum number of fine-tuning epochs. Defaults to 100. Keep "
        "in mind that early stopping is used."
    ),
    type=int,
)
@click.option(  # --batch-size
    "-bs",
    "--batch-size",
    default=64,
    help="Batch size. Defaults to 64.",
    type=int,
)
@click.option(  # --train-split
    "-ts",
    "--train-split",
    default="train",
    help="Name of the training data split in the dataset. Defaults to 'train'.",
    type=str,
)
@click.option(  # --val-split
    "-vs",
    "--val-split",
    default="val",
    help=(
        "Name of the validation data split in the dataset. Defaults to 'train'."
    ),
    type=str,
)
@click.option(  # --test-split
    "-es",
    "--test-split",
    default="test",
    help="Name of the test data split in the dataset. Defaults to 'train'.",
    type=str,
)
@click.option(  # --image-key
    "-ik",
    "--image-key",
    default="image",
    help="Image column name in the dataset. Defaults to 'image'.",
    type=str,
)
@click.option(  # --label-key
    "-lk",
    "--label-key",
    default="label",
    help="Label column name in the dataset. Defaults to 'label'.",
    type=str,
)
@click.option(  # --logit-key
    "-gk",
    "--logit-key",
    default="logits",
    help=(
        "Logit key in the model's output. Defaults to 'logits' which is the "
        "usual value."
    ),
    type=str,
)
@click.option(  # --head-name
    "-hn",
    "--head-name",
    default=None,
    help=(
        "Name of the model FC head submodule name, if it needs to be replaced "
        "(e.g. incorrect number of output neurons for the dataset). If "
        "specified, this name *must* point to a `nn.Linear` layer."
    ),
    type=str,
)
def train(
    batch_size: int,
    ce_weight: float,
    ckpt_path: Path | None,
    dataset_name: str,
    head_name: str | None,
    image_key: str,
    label_key: str,
    lcc_interval: int,
    lcc_k: int,
    lcc_submodules: str,
    lcc_warmup: int,
    lcc_weight: float,
    logit_key: str,
    max_epochs: int,
    model_name: str,
    output_dir: Path,
    test_split: str,
    train_split: str,
    val_split: str,
):
    """
    Performs latent cluster correction on a model fine-tuning using the
    `finetune` command.
    """
    import torch

    from .training import train as _train

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    _do_lcc = lcc_weight > 0 and lcc_interval > 0 and lcc_submodules
    _train(
        batch_size=batch_size,
        ce_weight=ce_weight,
        ckpt_path=ckpt_path,
        dataset_name=dataset_name,
        head_name=head_name,
        image_key=image_key,
        label_key=label_key,
        lcc_submodules=lcc_submodules.split(",") if _do_lcc else None,
        lcc_kwargs=(
            {
                "interval": lcc_interval,
                "warmup": lcc_warmup,
                "weight": lcc_weight,
                "k": lcc_k,
            }
            if _do_lcc
            else None
        ),
        logit_key=logit_key,
        max_epochs=max_epochs,
        model_name=model_name,
        output_dir=output_dir,
        test_split=test_split,
        train_split=train_split,
        val_split=val_split,
    )


@main.command()
@click.argument("model_name", type=str)
@click.option(  # --include-non-trainable
    "-i",
    "--include-non-trainable",
    help="Display non-trainable parameters.",
    is_flag=True,
)
@click.option(  # --max-depth
    "-d",
    "--max-depth",
    default=None,
    help="Maximum depth to display. Defaults to no limit.",
    type=int,
)
def pretty_print(model_name: str, include_non_trainable: bool, max_depth: int):
    """Pretty prints a HuggingFace model's structure"""

    from .utils import pretty_print_submodules

    if model_name.startswith("timm/"):
        import timm

        model = timm.create_model(model_name)
    else:
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(model_name)
    pretty_print_submodules(
        model,
        exclude_non_trainable=not include_non_trainable,
        max_depth=max_depth,
    )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
