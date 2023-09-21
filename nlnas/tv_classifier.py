"""
Wrapped classifier model from torchvision
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
from torch import nn
from torch.utils.hooks import RemovableHandle

from nlnas.utils import best_device


class TorchvisionClassifier(pl.LightningModule):
    """
    Classifier model from torchvision wrapped inside a `pl.LightningModule`
    """

    model: nn.Module
    """Internal model"""

    n_classes: int

    input_shape: list[int]
    """Shape of a **SINGLE** input (as opposed to the shape of a batch)"""

    def __init__(
        self,
        model_name: str,
        input_shape: list[int],
        n_classes: int = 10,
        weights: Any = None,
        **kwargs,
    ) -> None:
        """
        Args:
            model_name (str): See https://pytorch.org/vision/stable/models.html#classification
            input_shape (list[int]): Input shape of the image. For example,
                `CIFAR10` is `(3, 32, 32)`. If the number of channels is not
                `3`, an extra initial `1x1` convolution layer is added to
                transform the input images to 3-channels images.
            n_classes (int, optional):
            weights (Any, optional): See for example `https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.input_shape = input_shape
        self.n_classes = n_classes
        factory = getattr(torchvision.models, model_name)
        submodules = [factory(weights=weights), nn.LazyLinear(n_classes)]
        if input_shape[0] != 3:
            conv0 = nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            submodules = [conv0] + submodules
        self.model = nn.Sequential(*submodules)
        self.example_input_array = torch.zeros([1] + self.input_shape)
        self.model.eval()
        self.forward(self.example_input_array)

    def _evaluate(self, batch, stage: str | None = None) -> torch.Tensor:
        """Self-explanatory"""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y.long())
        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        if stage and best_device() != "mps":
            # NotImplementedError: The operator 'aten::_unique2' is not
            # currently implemented for the MPS device. If you want this op to
            # be added in priority during the prototype phase of this feature,
            # please comment on
            # https://github.com/pytorch/pytorch/issues/77764. As a temporary
            # fix, you can set the environment variable
            # `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for
            # this op. WARNING: this will be slower than running natively on
            # MPS.
            acc = torchmetrics.functional.accuracy(
                torch.argmax(logits, dim=1),
                y,
                "multiclass",
                num_classes=self.n_classes,
                top_k=1,
            )
            self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, *_, **__) -> torch.Tensor:
        return self.model(x.to(self.device))  # type: ignore

    def forward_intermediate(
        self,
        x: torch.Tensor,
        submodules: list[str],
        output_dict: dict,
    ) -> None:
        """
        Runs the model and collects the output of specified submodules. The
        intermediate outputs are stored in `output_dict` under the
        corresponding submodule name. In particular, this method has side
        effects.

        Args:
            x (torch.Tensor):
            submodules (list[str]):
            output_dict (dict):
        """

        def create_hook(key: str):
            def hook(
                _model: nn.Module, _args: Any, output: torch.Tensor
            ) -> None:
                output_dict[key] = output.detach().cpu()

            return hook

        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        self(x)
        for h in handles:
            h.remove()

    def test_step(self, batch, *_, **__):
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch, *_, **__) -> Any:
        x, y = batch
        loss = nn.functional.cross_entropy(self.model(x), y.long())
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, *_, **__):
        return self._evaluate(batch, "val")
