"""Import required losses."""


from .classification_loss_functions import (
    WeightedCrossEntropyLoss,
    WeightedFocalLoss,
)

__all__ = [
    "WeightedCrossEntropyLoss",
    "WeightedFocalLoss"
]