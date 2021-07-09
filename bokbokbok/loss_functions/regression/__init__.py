"""Import required losses."""


from .regression_loss_functions import (
    LogCoshLoss,
    SPELoss,
)

__all__ = [
    "LogCoshLoss",
    "SPELoss",
]