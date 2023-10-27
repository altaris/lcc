"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from .classifier import (
    Classifier,
    TorchvisionClassifier,
    TruncatedClassifier,
    VHTorchvisionClassifier,
)
from .nlnas import analyse_ckpt, analyse_training, train_and_analyse_all
from .pdist import pdist
from .plotting import class_scatter, gaussian_mixture_plot
from .separability import gdv, label_variation, pairwise_svc_scores
from .training import (
    all_checkpoint_paths,
    best_epoch,
    train_model,
    train_model_guarded,
)
from .tv_dataset import TorchvisionDataset
from .utils import best_device, dataset_n_targets, get_first_n
