"""Everything related to latent clustering correction"""

from .choice import max_connected_confusion_choice
from .clustering import (
    class_otm_matching,
    lcc_knn_indices,
    lcc_loss,
    lcc_targets,
    otm_matching_predicates,
)
from .louvain import louvain_communities
