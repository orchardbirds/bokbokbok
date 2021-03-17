"""Import required losses."""


from .regression_loss_functions import (
    SquaredLogErrorLoss,
    LogCoshLoss,
)

__all__ = [
    "SquaredLogErrorLoss",
    "LogCoshLoss"
]