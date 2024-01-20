from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as logging
from torchvision.models import AlexNet_Weights, ResNet18_Weights

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.training import train_model_guarded
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    experiments = [
        {
            "model_name": "resnet18",
            "weights": ResNet18_Weights.IMAGENET1K_V1,
            "correction_submodules": [
                "model.0.layer3",
                "model.0.layer4",
                # "model.0.fc",
            ],
        },
        {
            "model_name": "alexnet",
            "weights": AlexNet_Weights.IMAGENET1K_V1,
            "correction_submodules": [
                "model.0.classifier.1",
                "model.0.classifier.4",
                # "model.0.classifier.6",
            ],
        },
        # {
        #     "model_name": "vit_b_16",
        #     "weights": ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        #     "correction_submodules": [
        #         # "model.0.encoder.layers.encoder_layer_0.mlp",
        #         # "model.0.encoder.layers.encoder_layer_1.mlp",
        #         # "model.0.encoder.layers.encoder_layer_2.mlp",
        #         # "model.0.encoder.layers.encoder_layer_3.mlp",
        #         # "model.0.encoder.layers.encoder_layer_4.mlp",
        #         # "model.0.encoder.layers.encoder_layer_5.mlp",
        #         # "model.0.encoder.layers.encoder_layer_6.mlp",
        #         # "model.0.encoder.layers.encoder_layer_7.mlp",
        #         # "model.0.encoder.layers.encoder_layer_8.mlp",
        #         # "model.0.encoder.layers.encoder_layer_9.mlp",
        #         "model.0.encoder.layers.encoder_layer_10.mlp",
        #         "model.0.encoder.layers.encoder_layer_11.mlp",
        #         # "model.0.heads",
        #     ],
        # },
    ]
    weight_exponent, batch_size, k = 3, 2048, 5
    for exp in experiments:
        try:
            exp_name = (
                exp["model_name"]
                + f"_finetune_l{k}_b{batch_size}_1e-{weight_exponent}"
            )
            output_dir = Path("out") / exp_name / "imagenet"
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = batch_size
            datamodule = TorchvisionDataset(
                "imagenet",
                transform=exp["weights"].transforms,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                exp["model_name"],
                n_classes=1000,
                model_config={"weights": exp["weights"]},
                cor_type="louvain",
                cor_weight=10 ** (-weight_exponent),
                cor_submodules=exp["correction_submodules"],
                cor_kwargs={"k": k},
            )
            model = model.to(best_device())
            train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=exp_name,
                max_epochs=512,
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=exp_name,
            # )
        except (KeyboardInterrupt, SystemExit):
            return
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()