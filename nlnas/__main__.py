"""CLI module"""

import os
from pathlib import Path

import click
from loguru import logger as logging


@click.group()
@click.option(
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
@click.argument(
    "result_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),  # type: ignore
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),  # type: ignore
)
@click.argument("correction_submodules", type=str)
@click.argument("correction_weight", type=float)
@click.option(
    "-e",
    "--max-epochs",
    default=100,
    help=(
        "Maximum number of fine-tuning epochs. Defaults to 100. Keep "
        "in mind that early stopping is used."
    ),
    type=int,
)
@click.option(
    "-bs",
    "--batch-size",
    default=64,
    help="Batch size. Defaults to 64.",
    type=int,
)
@click.option(
    "-nc",
    "--n-classes-per-batch",
    default=5,
    help="Number of classes to be represented in each batch.",
    type=int,
)
def correct(
    result_file: Path,
    output_dir: Path,
    correction_submodules: str,
    correction_weight: float,
    max_epochs: int,
    batch_size: int,
    n_classes_per_batch: int,
):
    """
    Performs latent cluster correction on a model fine-tuning using the
    `finetune` command.
    """
    import torch

    from .finetune import correct as _correct

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    _correct(
        result_file=result_file,
        output_dir=output_dir,
        correction_weight=correction_weight,
        correction_submodules=correction_submodules.split(","),
        max_epochs=max_epochs,
        batch_size=batch_size,
        n_classes_per_batch=n_classes_per_batch,
    )


@main.command()
@click.argument("model_name", type=str)
@click.argument("dataset_name", type=str)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),  # type: ignore
)
@click.option(
    "-e",
    "--max-epochs",
    default=100,
    help=(
        "Maximum number of fine-tuning epochs. Defaults to 100. Keep "
        "in mind that early stopping is used."
    ),
    type=int,
)
@click.option(
    "-bs",
    "--batch-size",
    default=64,
    help="Batch size. Defaults to 64.",
    type=int,
)
@click.option(
    "-ts",
    "--train-split",
    default="train",
    help="Name of the training data split in the dataset. Defaults to 'train'.",
    type=str,
)
@click.option(
    "-vs",
    "--val-split",
    default="val",
    help=(
        "Name of the validation data split in the dataset. Defaults to 'train'."
    ),
    type=str,
)
@click.option(
    "-es",
    "--test-split",
    default="test",
    help="Name of the test data split in the dataset. Defaults to 'train'.",
    type=str,
)
@click.option(
    "-ik",
    "--image-key",
    default="image",
    help="Image column name in the dataset. Defaults to 'image'.",
    type=str,
)
@click.option(
    "-lk",
    "--label-key",
    default="label",
    help="Label column name in the dataset. Defaults to 'label'.",
    type=str,
)
@click.option(
    "-gk",
    "--logit-key",
    default="logits",
    help=(
        "Logit key in the model's output. Defaults to 'logits' which is the "
        "usual value."
    ),
    type=str,
)
@click.option(
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
@click.option(
    "-cw",
    "--correction-weight",
    default=0,
    help=(
        "Latent clustering correction weight. Defaults to 0 (meaning no "
        "correction is applied)"
    ),
    type=float,
)
@click.option(
    "-cs",
    "--correction-submodules",
    default="",
    help=(
        "Command-separated list of submodule names whose output space will "
        "undergo latent clustering correction. Defaults to empty string "
        "(meaning no correction is applied)."
    ),
    type=str,
)
def finetune(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    max_epochs: int,
    batch_size: int,
    train_split: str,
    val_split: str,
    test_split: str,
    image_key: str,
    label_key: str,
    logit_key: str,
    head_name: str | None,
    correction_weight: float,
    correction_submodules: str,
):
    """Fine-tune a HuggingFace model on a HuggingFace dataset."""
    import torch

    from .finetune import finetune as _finetune

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    _finetune(
        model_name=model_name,
        dataset_name=dataset_name,
        output_dir=output_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        head_name=head_name,
        correction_weight=correction_weight,
        correction_submodules=(
            correction_submodules.split(",") if correction_submodules else None
        ),
    )


@main.command()
@click.argument("model_name", type=str)
@click.option(
    "-i",
    "--include-non-trainable",
    help="Display non-trainable parameters.",
    is_flag=True,
)
@click.option(
    "-d",
    "--max-depth",
    default=None,
    help="Maximum depth to display. Defaults to no limit.",
    type=int,
)
def pretty_print(model_name: str, include_non_trainable: bool, max_depth: int):
    """Pretty prints a HuggingFace model's structure"""
    from transformers import AutoModelForImageClassification

    from .utils import pretty_print_submodules

    model = AutoModelForImageClassification.from_pretrained(model_name)
    pretty_print_submodules(
        model,
        exclude_non_trainable=not include_non_trainable,
        max_depth=max_depth,
    )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
