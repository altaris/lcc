from datetime import datetime

import pytorch_lightning as pl
import torch
from torchvision.models import ResNet18_Weights, AlexNet_Weights

from nlnas.classifier import TorchvisionClassifier
from nlnas.imagenet import ImageNet

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    start = datetime.now()
    ds = ImageNet(
        transform=AlexNet_Weights.DEFAULT.transforms(),
        download_path="/home/cedric/torchvision/imagenet/",
    )
    model = TorchvisionClassifier(
        "alexnet",
        n_classes=1000,
        model_config={"weights": AlexNet_Weights.DEFAULT},
    )
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.TQDMProgressBar()],
        strategy="ddp",
        use_distributed_sampler=False,
    )
    data = trainer.test(model, ds)
    print(data)
    print("Done in", datetime.now() - start)
