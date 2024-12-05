import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from loguru import logger as logging
from pytorch_lightning.strategies import ParallelStrategy

from lcc.classifiers import HuggingFaceClassifier
from lcc.datasets import HuggingFaceDataset


HF_MODEL_NAME = "microsoft/resnet-18"
DATASET = "cifar100"
TRAIN_SPLIT = "train"
VAL_SPLIT = "train[80%:]"
TEST_SPLIT = "test"
IMAGE_KEY = "img"
LABEL_KEY = "fine_label"


def all_gather_concat(strategy: ParallelStrategy, x: torch.Tensor) -> Tensor:
    """
    `all_gather`s and concatenate 2D tensors across all ranks. `x` is expected
    to have shape `(n, d)`, where `d` is the same across ranks, but where `n`
    can vary (as opposed to `Fabric.all_gather`, where all tensors are truncated
    to match the length of the shortest one).
    """

    _all_gather = strategy.all_gather
    lengths = _all_gather(torch.tensor(len(x)).to(x))
    lengths = lengths.int().tolist()
    logging.info("[RANK {}] lengths {}", strategy.global_rank, lengths)
    x = torch.nn.functional.pad(x, (0, 0, 0, max(lengths) - len(x)))
    all_x = _all_gather(x)
    logging.info(
        "[RANK {}] all_x shapes {}",
        strategy.global_rank,
        [u.shape for u in all_x],
    )
    logging.info(
        "[RANK {}] all_x tails {}",
        strategy.global_rank,
        [u[-2:] for u in all_x],
    )
    x = torch.cat([t[:s] for t, s in zip(all_x, lengths) if s > 0])
    assert len(x) == sum(lengths)  # FOR DEBUGGING
    return x


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.fc = torch.nn.Linear(512, 100)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x = batch[IMAGE_KEY].flatten(1)[:, :512]
        logging.info("[RANK {}] x {}", self.global_rank, x.shape)
        # b = self.trainer.strategy.all_gather(x)
        b = all_gather_concat(
            self.trainer.strategy,
            x[self.trainer.global_rank :],
        )
        logging.info("[RANK {}] b {}", self.global_rank, b.shape)
        return nn.functional.cross_entropy(self.fc(x), batch[LABEL_KEY])


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy="ddp",
        max_epochs=1,
        default_root_dir="out.test/lightning_logs",
    )

    image_processor = HuggingFaceClassifier.get_image_processor(HF_MODEL_NAME)
    ds = HuggingFaceDataset(
        dataset_name=DATASET,
        fit_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        label_key=LABEL_KEY,
        image_processor=image_processor,
    )

    model = Model()
    trainer.fit(model, ds)
