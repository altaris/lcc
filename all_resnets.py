from itertools import product

from loguru import logger as logging

from nlnas.nlnas import training_suite


def main():
    model_names = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]
    submodule_names = "maxpool,layer1,layer2,layer3,layer4,fc"
    dataset_names = [
        # "mnist",
        # "kmnist",
        # "fashionmnist",
        # "cifar10",
        "cifar100",
    ]
    for m, d in product(model_names, dataset_names):
        training_suite(
            model_name=m,
            submodule_names=submodule_names,
            dataset_name=d,
            output_dir="export-out",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
