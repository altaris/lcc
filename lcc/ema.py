"""
Exponential Moving Average (EMA) callack for pytorch lightning. Code taken and
updated from
https://colab.research.google.com/github/benihime91/gale/blob/master/nbs/07b_collections.callbacks.ema.ipynb
"""

from typing import Any, Iterable

import pytorch_lightning as pl
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV3
from torch import Tensor

CKPT_KEY = "state_dict_ema"
"""
Key to use when saving/loading a checkpoint dict.
"""


class EMACallback(pl.Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using
    the moving average of the trained parameters of a deep network is better
    than using its trained parameters directly.
    """

    decay: float
    ema: ModelEmaV3
    use_ema_weights: bool

    def __init__(self, decay: float = 0.9999, use_ema_weights: bool = True):
        """
        Args:
            decay (float, optional):
            use_ema_weights (bool, optional): If `True`, then the ema parameters
                of the network are set onto the original model after training
                end.
        """
        self.decay, self.use_ema_weights = decay, use_ema_weights

    def copy_to(
        self, shadow_parameters: Iterable[Tensor], parameters: Iterable[Tensor]
    ) -> None:
        """Copy current parameters into given collection of parameters."""
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.ema = ModelEmaV3(pl_module, decay=self.decay, device=None)
        super().on_fit_start(trainer, pl_module)

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema is not None:
            self.ema.module.load_state_dict(checkpoint[CKPT_KEY])
        super().on_load_checkpoint(trainer, pl_module, checkpoint)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema is not None:
            checkpoint[CKPT_KEY] = get_state_dict(self.ema, unwrap_model)
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.ema.update(pl_module)
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        super().on_train_end(trainer, pl_module)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.restore(pl_module.parameters())
        super().on_validation_end(trainer, pl_module)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.store(pl_module.parameters())
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        super().on_validation_start(trainer, pl_module)

    def restore(self, parameters: Iterable[Tensor]) -> None:
        """
        Restore the parameters stored with the `store` method.  Useful to
        validate the model with EMA parameters without affecting the original
        optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def store(self, parameters: Iterable[Tensor]) -> None:
        """Save the current parameters for restoring later."""
        self.collected_params = [param.clone() for param in parameters]
