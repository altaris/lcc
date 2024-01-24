from datetime import datetime

import pytorch_lightning as pl
import torch
from torchvision.models import ResNet18_Weights

from nlnas.classifier import TorchvisionClassifier
from nlnas.imagenet import ImageNet

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    start = datetime.now()
    ds = ImageNet(
        transform=ResNet18_Weights.DEFAULT.transforms(),
        download_path="/home/cedric/torchvision/imagenet/",
    )
    model = TorchvisionClassifier(
        "resnet18", n_classes=1000, model_config={"weights": "DEFAULT"}
    )
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.TQDMProgressBar()],
        strategy="ddp",
    )
    data = trainer.test(model, ds)
    print(data)
    print("Done in", datetime.now() - start)
