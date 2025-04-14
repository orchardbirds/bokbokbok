import numpy as np
from scipy.special import expit


# def clip_sigmoid(yhat):
#     """
#     Applies the sigmoid function and ensures that the values lie in the range
#     1e-15 < yhat < 1 - 1e-15.
#     We clip to avoid dividing by zero in the loss functions.

#     Args:
#         yhat: The margin probabilities yet to be put into a sigmoid function

#     Returns:
#         yhat: The clipped probabilities
#     """
#     yhat = 1. / (1. + np.exp(-yhat))
#     yhat[yhat >= 1] = 1 - 1e-6
#     yhat[yhat <= 0] = 1e-6
#     return yhat

def clip_sigmoid(yhat, eps=1e-15, library="XGBoost"):
    if library != "XGBoost":
        return np.clip(expit(yhat), eps, 1 - eps)
    return np.clip(yhat, eps, 1 - eps)
