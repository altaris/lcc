"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from .analysis import analyse_ckpt, analyse_training
from .classifiers import (
    BaseClassifier,
    HuggingFaceClassifier,
    TorchvisionClassifier,
    WrappedClassifier,
)
from .correction import (
    class_otm_matching,
    clustering_loss,
    louvain_communities,
    louvain_loss,
    max_connected_confusion_choice,
    otm_matching_predicates,
)
from .datasets import (
    DEFAULT_DATALOADER_KWARGS,
    HuggingFaceDataset,
    TorchvisionDataset,
    WrappedDataset,
    dl_head,
)
from .plotting import class_scatter, gaussian_mixture_plot
from .training import all_checkpoint_paths, best_checkpoint_path, best_epoch
from .utils import best_device
