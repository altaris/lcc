from itertools import product
from pathlib import Path

import torchvision
from loguru import logger as logging

from nlnas import Classifier, TensorDataset, train_and_analyse_all


def main():
    model_names = [
        "vit_b_16",
    ]
    submodule_names = [
        "model.0.encoder.layers.encoder_layer_0",
        "model.0.encoder.layers.encoder_layer_1",
        "model.0.encoder.layers.encoder_layer_2",
        "model.0.encoder.layers.encoder_layer_3",
        "model.0.encoder.layers.encoder_layer_4",
        "model.0.encoder.layers.encoder_layer_5",
        "model.0.encoder.layers.encoder_layer_6",
        "model.0.encoder.layers.encoder_layer_7",
        "model.0.encoder.layers.encoder_layer_8",
        "model.0.encoder.layers.encoder_layer_9",
        "model.0.encoder.layers.encoder_layer_10",
        "model.0.encoder.layers.encoder_layer_11",
        "model.0.heads",
        "model.1",
    ]
    dataset_names = [
        "mnist",
        # "kmnist",
        # "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([224, 224], antialias=True),
            # torchvision.transforms.Normalize([0], [1]),
        ]
    )
    for m, d in product(model_names, dataset_names):
        output_dir = Path("export-out") / m / d
        ds = TensorDataset.from_torchvision_dataset(d, transform)
        # ds.x, ds.y = ds.x[:512], ds.y[:512]
        if ds.x.shape[1] != 3:
            logging.debug("Converting the image dataset to RGB")
            ds.x = ds.x.repeat(1, 3, 1, 1)
        model = Classifier(
            model_name=m,
            n_classes=ds.n_classes,
            add_final_fc=True,
            input_shape=ds.image_shape,
        )
        train_and_analyse_all(
            model=model,
            submodule_names=submodule_names,
            dataset=ds,
            output_dir=output_dir,
            model_name=m,
        )


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     pass
    # except:
    #     logging.exception(":sad trombone:")
