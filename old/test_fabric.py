import os
from time import sleep

import torch
from lightning_fabric import Fabric
from loguru import logger as logging


N_DEVICES = os.getenv("N_DEVICES", 1)


def f(x: int, fabric) -> int:
    if fabric.global_rank == 0:
        x += 1
        logging.info("[RANK {}] Inside f, seeping", fabric.global_rank)
        sleep(10)
        logging.info("[RANK {}] Inside f, waking up", fabric.global_rank)
    x = fabric.broadcast(x, src=0)
    return x


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    fabric = Fabric(devices=N_DEVICES)
    fabric.launch()
    logging.info("[RANK {}] Started", fabric.global_rank)

    logging.info("[RANK {}] Before calling f", fabric.global_rank)
    x = f(0, fabric)
    logging.info("[RANK {}] After calling f, result={}", fabric.global_rank, x)
