from datetime import datetime

import numpy as np
from loguru import logger as logging

from nlnas.classifiers import HuggingFaceClassifier
from nlnas.correction import louvain_communities
from nlnas.datasets import HuggingFaceDataset

BATCH_SIZE = 1024
PCA_N_COMPONENTS = 1024

logging.info("Loading dataset")
dataset = HuggingFaceDataset(
    "cifar100",
    fit_split="train[:5%]",
    val_split="train[80%:]",
    test_split="test",
    image_processor=HuggingFaceClassifier.get_image_processor(
        "microsoft/resnet-18"
    ),
    train_dl_kwargs={"batch_size": BATCH_SIZE, "drop_last": True},
)
IMAGE_KEY = "img"
# dataset = HuggingFaceDataset(
#     "ILSVRC/imagenet-1k",
#     fit_split="train",
#     val_split="validation",
#     test_split="test",
#     image_processor=HuggingFaceClassifier.get_image_processor(
#         "microsoft/resnet-18"
#     ),
#     train_dl_kwargs={"batch_size": BATCH_SIZE, "drop_last": True},
# )
# IMAGE_KEY = "image"
dataset.setup("fit")

logging.info("Finding Louvain communities")
start = datetime.now()

communities, _ = louvain_communities(
    dataset.train_dataloader(),
    key=IMAGE_KEY,
    pca_dim=PCA_N_COMPONENTS,
    k=10,
    device="cuda",
    tqdm_style="console",
)

logging.info("Found Louvain communities in {}", datetime.now() - start)
logging.debug("Found {} communities", len(communities))
logging.debug(
    "Community size: min={}, max={}, med={}",
    min(len(c) for c in communities),
    max(len(c) for c in communities),
    np.median([len(c) for c in communities]),
)
