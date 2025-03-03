"""
LCC losses flavors.

LCC is all about detecting and correcting misclustered samples. Detection occurs
at the clustering stage, see `lcc.correction.clustering` and specifically
`lcc.correction.clustering.otm_matching_predicates`.

Then comes the question of correction. Misclustered samples are pulled towards a
_target_ that is itself correctly clustered (and in the same class). This
package contains some loss object that do just that.

The current loss implementations are:
* `lcc.correction.loss.ExactLCCLoss` which is the theoretically exact loss
* `lcc.correction.loss.RandomizedLCCLoss` which is a randomized approximation of
  the exact loss at a fraction of the computational cost.
"""

from .base import LCCLoss
from .exact import ExactLCCLoss
from .randomized import RandomizedLCCLoss

__all__ = [
    "ExactLCCLoss",
    "LCCLoss",
    "RandomizedLCCLoss",
]
