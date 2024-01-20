from datetime import datetime

import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights

from nlnas.tv_dataset import TorchvisionDataset
from nlnas.classifier import TorchvisionClassifier


if __name__ == "__main__":
    start = datetime.now()
    ds = TorchvisionDataset(
        "imagenet",
        download_path="/home/cedric/torchvision/datasets/imagenet/",
        transform=ResNet18_Weights.DEFAULT.transforms(),
    )
    ds.setup("test")
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
