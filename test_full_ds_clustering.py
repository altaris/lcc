from pathlib import Path

import turbo_broccoli as tb

from nlnas import HuggingFaceClassifier, HuggingFaceDataset
from nlnas.correct import full_dataset_latent_clustering

HF_MODEL_NAME = "microsoft/resnet-18"
CKPT_PATH = "out/ft/cifar100/microsoft-resnet-18/tb_logs/microsoft-resnet-18/version_0/checkpoints/epoch=14-step=4695.ckpt"
LOGIT_KEY = "logits"
HEAD_NAME = "classifier.1"

CORRECTION_SUBMODULES = [  # See also `nlnas.utils.pretty_print_submodules`
    "resnet.encoder.stages.0",
    "resnet.encoder.stages.1",
    "resnet.encoder.stages.2",
    "resnet.encoder.stages.3",
]
CORRECTION_WEIGHT = 1e-5

K = 5

HF_DATASET_NAME = "cifar100"
TRAIN_SPLIT = "train[:80%]"
VAL_SPLIT = "train[80%:]"
TEST_SPLIT = "test"
IMAGE_KEY = "img"
LABEL_KEY = "fine_label"

DATASET_NAME = HF_DATASET_NAME.replace("/", "-")
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")
OUTPUT_DIR = Path("out.test") / DATASET_NAME / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    dataset = HuggingFaceDataset(
        dataset_name=HF_DATASET_NAME,
        fit_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        label_key=LABEL_KEY,
        image_processor=HF_MODEL_NAME,
    )
    dataset.setup("fit")
    model = HuggingFaceClassifier(
        model_name=HF_MODEL_NAME,
        n_classes=dataset.n_classes(),
        head_name=HEAD_NAME,
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        logit_key=LOGIT_KEY,
        cor_submodules=CORRECTION_SUBMODULES,
        cor_weight=CORRECTION_WEIGHT,
    )
    # pylint: disable=no-value-for-parameter
    model.model = HuggingFaceClassifier.load_from_checkpoint(CKPT_PATH).model
    data = full_dataset_latent_clustering(model, dataset, OUTPUT_DIR, k=K)
    tb.save_json(data, OUTPUT_DIR / "data.json")
