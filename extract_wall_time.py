"""
Extracts the training time (for each epoch) for all `tfevents` files located in
`out/` (or subfolders thereof). Run this with python3.11.

The results are written to `out/wall.csv` which looks like this:

    ,wall_time,step,model,dataset,weight,k,epoch
    0,0.0,4,alexnet,cifar10,0.0,0,0
    1,0.4084627628326416,9,alexnet,cifar10,0.0,0,0
    2,0.7298846244812012,14,alexnet,cifar10,0.0,0,0
    3,1.064856767654419,19,alexnet,cifar10,0.0,0,0
    4,2.1463263034820557,19,alexnet,cifar10,0.0,0,0
    5,5.541053771972656,24,alexnet,cifar10,0.0,0,1
    6,6.058325290679932,29,alexnet,cifar10,0.0,0,1
    ...

"""

from pathlib import Path
from pyexpat import model

import pandas as pd
import regex as re
from loguru import logger as logging
from tensorboard.backend.event_processing import event_accumulator

OUTPUT_DIR = Path("out")
train_re = (
    r".*out/([^/]+)/(\w+)/model/tb_logs/[^/]+/version_0/events.out.tfevents.*"
)
ftune_re = r"(\w+)_finetune_l(\d+)_b\d+_1e-(\d+)"

dfs = []
for path in OUTPUT_DIR.rglob("events.out.tfevents.*"):
    if m := re.match(train_re, str(path)):
        try:
            experiment, dataset = m.group(1), m.group(2)
            if m := re.match(ftune_re, experiment):
                model, k, weight = (
                    m.group(1),
                    int(m.group(2)),
                    10 ** (-int(m.group(3))),
                )
                logging.info(
                    "Found fine-tuning: model={}, ds={}, k={} w={}",
                    model,
                    dataset,
                    k,
                    weight,
                )
            else:
                model, k, weight = experiment, 0, 0
                logging.info(
                    "Found pretraining: model={}, ds={}",
                    model,
                    dataset,
                )
            ea = event_accumulator.EventAccumulator(str(path))
            ea.Reload()
            df = pd.DataFrame(ea.Scalars("epoch"))
            df["model"], df["dataset"] = model, dataset
            # df["k"], df["weight"] = k, weight
            df["weight"], df["k"] = weight, k
            df["wall_time"] = df["wall_time"] - df["wall_time"].min()
            df["epoch"] = df["value"].astype(int)
            df.drop("value", axis=1, inplace=True)
            dfs.append(df)
        except Exception as e:
            logging.error("{} {}", type(e), str(e))
    else:
        logging.error("Skipping event file {}", path)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(OUTPUT_DIR / "wall.csv")
