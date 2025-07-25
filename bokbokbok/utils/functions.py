import numpy as np

from typing import Any

# Typing needs to be made more specific for different matrices, but w/e
def clip_sigmoid(yhat: Any) -> np.ndarray:
    """
    Applies the sigmoid function and ensures that the values lie in the range
    1e-15 < yhat < 1 - 1e-15.
    We clip to avoid dividing by zero in the loss functions.

    Args:
        yhat: The margin probabilities yet to be put into a sigmoid function

    Returns:
        yhat: The clipped probabilities
    """
    yhat = 1. / (1. + np.exp(-yhat))
    yhat[yhat >= 1] = 1 - 1e-15
    yhat[yhat <= 0] = 1e-15
    return yhat
