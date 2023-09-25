from itertools import product
from pathlib import Path

from loguru import logger as logging

from nlnas import Classifier, TensorDataset, train_and_analyse_all


def main():
    model_names = [
        "googlenet",
    ]
    submodule_names = [
        "model.0.maxpool1",
        "model.0.maxpool2",
        "model.0.inception3a",
        "model.0.maxpool3",
        "model.0.inception4a",
        "model.0.inception4b",
        "model.0.inception4c",
        "model.0.inception4d",
        "model.0.maxpool4",
        "model.0.inception5a",
        "model.0.inception5b",
        "model.0.fc",
        "model.1",
    ]
    dataset_names = [
        "mnist",
        # "kmnist",
        "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    for m, d in product(model_names, dataset_names):
        output_dir = Path("export-out") / m / d
        ds = TensorDataset.from_torchvision_dataset(d)
        if ds.x.shape[1] != 3:
            logging.debug("Converting the image dataset to RGB")
            ds.x = ds.x.repeat(1, 3, 1, 1)
        train_and_analyse_all(
            model=Classifier.torchvision_classifier(m, n_classes=ds.n_classes),
            submodule_names=submodule_names,
            dataset=ds,
            output_dir=output_dir,
            model_name=m,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
