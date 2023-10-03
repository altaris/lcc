from pathlib import Path

import torch
from loguru import logger as logging
from torch import Tensor, nn

from nlnas import TorchvisionDataset
from nlnas.classifier import Classifier


class Zero(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class Cell(nn.Module):
    OPERATIONS = {
        "zero": Zero,
    }
    edges: nn.ModuleDict
    weights: nn.ParameterDict

    def __init__(self, n_nodes: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        pass


def main():
    output_dir = Path("out") / "nas"
    ds = TorchvisionDataset("cifar10")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
