"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from .classifiers import (
    BaseClassifier,
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
    dl_targets,
)
from .datasets.transforms import EnsureRGB, dataset_normalization
from .nlnas import analyse_ckpt, analyse_training, train_and_analyse_all
from .plotting import class_scatter, gaussian_mixture_plot
from .training import (
    all_checkpoint_paths,
    best_checkpoint_path,
    best_epoch,
    train_model,
    train_model_guarded,
)
from .utils import best_device
