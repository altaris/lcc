import sys
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as logging
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)

from nlnas import (
    ImageNet,
    TorchvisionClassifier,
    best_device,
    train_model_guarded,
)
from nlnas.logging import r0_info, setup_logging

IMAGENET_DOWNLOAD_PATH = Path.home() / "torchvision" / "imagenet"


def main():
    pl.seed_everything(0)
    experiments = [
        {
            "model_name": "resnet18",
            "weights": ResNet18_Weights.DEFAULT,
            "correction_submodules": [
                "model.0.layer3",
                "model.0.layer4",
                "model.0.fc",
            ],
            # torch.optim.SGD(self.parameters(), lr=1e-5, momentum=0.9)
            # test/loss: 1.246917963027954, test/acc: 0.6976400017738342
        },
        # {
        #     "model_name": "alexnet",
        #     "weights": AlexNet_Weights.DEFAULT,
        #     "correction_submodules": [
        #         "model.0.classifier.1",
        #         "model.0.classifier.4",
        #         "model.0.classifier.6",
        #     ],
        #     # torch.optim.SGD(self.parameters(), lr=1e-5, momentum=0.9)
        #     # test/loss: 1.9095762968063354, test/acc: 0.565500020980835
        # },
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
    weight_exponent, batch_size, k = 1, 2048, 10
    for exp in experiments:
        try:
            r0_info(
                "Starting fine-tuning of {} with k={} and w=1e-{}",
                exp["model_name"],
                batch_size,
                weight_exponent,
            )
            exp_name = (
                exp["model_name"]
                + f"_finetune_l{k}_b{batch_size}_1e-{weight_exponent}"
            )
            output_dir = Path("out") / exp_name / "imagenet"
            datamodule = ImageNet(
                transform=exp["weights"].transforms(),
                download_path=IMAGENET_DOWNLOAD_PATH,
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
            model, _ = train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=exp_name,
                max_epochs=128,
                use_distributed_sampler=False,
                early_stopping_kwargs={
                    "monitor": "val/loss",
                    "patience": 10,
                    "mode": "min",
                },
            )
            if model.global_rank != 0:
                sys.exit(0)
            r0_info("Testing validation accuracy")
            trainer = pl.Trainer(
                callbacks=[pl.callbacks.TQDMProgressBar()],
                strategy="ddp",
                use_distributed_sampler=False,
            )
            trainer.test(model, datamodule)
        except (KeyboardInterrupt, SystemExit):
            return
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
