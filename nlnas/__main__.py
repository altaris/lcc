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
@click.argument("model_name", type=str)
@click.argument("dataset_name", type=str)
@click.argument("n_classes", type=int)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),  # type: ignore
)
@click.option(
    "-e",
    "--epochs",
    default=10,
    help="Number of fine-tuning epochs. Defaults to 10.",
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
def finetune(
    model_name: str,
    dataset_name: str,
    n_classes: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    train_split: str,
    val_split: str,
    test_split: str,
    image_key: str,
    label_key: str,
    logit_key: str,
    head_name: str | None,
):
    """Fine-tune a HuggingFace model on a HuggingFace dataset."""
    import torch

    from .finetune import finetune as _finetune

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    _finetune(
        model_name=model_name,
        dataset_name=dataset_name,
        n_classes=n_classes,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        head_name=head_name,
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
