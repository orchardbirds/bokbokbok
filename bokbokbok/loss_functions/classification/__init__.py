"""Import required losses."""


from .classification_loss_functions import (
    WeightedCrossEntropyLoss,
    FocalLoss,
)

__all__ = [
    "WeightedCrossEntropyLoss",
    "FocalLoss"
]