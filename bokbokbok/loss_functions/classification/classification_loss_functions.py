import numpy as np
from bokbokbok.utils import clip_sigmoid


def WeightedCrossEntropyLoss(alpha=0.5):
    """
    Calculates the Weighted Cross-Entropy Loss, which applies
    a factor alpha to the regular Cross-Entropy Loss.
    """

    def _gradient(yhat, dtrain, alpha):
        """Compute the weighted cross-entropy gradient.

        Args:
            yhat (np.array): Margin predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            grad: Weighted cross-entropy gradient
        """
        y = dtrain.get_label()

        yhat = clip_sigmoid(yhat)

        grad = yhat - (y * alpha)

        return grad

    def _hessian(yhat):
        """Compute the weighted cross-entropy hessian.

        Args:
            yhat (np.array): Margin predictions

        Returns:
            hess: Weighted cross-entropy Hessian
        """

        yhat = clip_sigmoid(yhat)

        hess = yhat * (1 - yhat)

        return hess

    def weighted_cross_entropy(
            yhat,
            dtrain,
            alpha=alpha
    ):
        """
        Calculate gradient and hessian for weight cross-entropy,

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            grad: Weighted cross-entropy gradient
            hess: Weighted cross-entropy Hessian
        """
        grad = _gradient(yhat, dtrain, alpha=alpha)

        hess = _hessian(yhat)

        return grad, hess

    return weighted_cross_entropy


def FocalLoss(alpha=1.0, gamma=2.0):
    """
    Calculates the Weighted Focal Loss, see
    https://arxiv.org/pdf/1708.02002.pdf
    Note that if using alpha =1 and gamma = 0,
    this is the same as using regular Cross Entropy

    """

    def _gradient(yhat, dtrain, alpha, gamma):
        """Compute the focal gradient.

        Args:
            yhat (np.array): Margin predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter

        Returns:
            grad: Weighted Focal Loss gradient
        """
        y = dtrain.get_label()

        yhat = clip_sigmoid(yhat)

        L1 = alpha * gamma * y * yhat * np.log(yhat) * np.power((1 - yhat), gamma)
        L2 = -1. * alpha * y * np.power((1 - yhat), gamma + 1)
        L3 = -1. * gamma * (1 - y) * np.power(yhat, gamma) * np.log(1 - yhat) * (1 - yhat)
        L4 = (1 - y) * np.power(yhat, gamma + 1)

        grad = L1 + L2 + L3 + L4

        return grad

    def _hessian(yhat, dtrain, alpha, gamma):
        """Compute the weighted cross-entropy hessian.

        Args:
            yhat (np.array): Margin predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter

        Returns:
            hess: Weighted Focal Loss Hessian
        """
        y = dtrain.get_label()

        yhat = clip_sigmoid(yhat)

        L1 = alpha * gamma * y * yhat * np.power(1 - yhat, gamma + 1) * np.log(yhat)
        L2 = alpha * gamma * y * yhat * np.power(1 - yhat, gamma + 1)
        L3 = -1. * alpha * np.power(gamma, 2) * y * np.power(yhat, 2) * np.power(1 - yhat, gamma) * np.log(yhat)
        L4 = alpha * y * (gamma + 1) * yhat * np.power(1 - yhat, gamma + 1)
        L5 = -1. * np.power(gamma, 2) * (1 - y) * np.power(yhat, gamma) * np.log(1 - yhat) * np.power(1 - yhat, 2)
        L6 = gamma * (1 - y) * np.power(yhat, gamma + 1) * (1 - yhat)
        L7 = gamma * (1 - y) * np.power(yhat, gamma + 1) * np.log(1 - yhat) * (1 - yhat)
        L8 = (gamma + 1) * (1 - y) * np.power(yhat, gamma + 1) * (1 - yhat)

        hess = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8
        return hess

    def focal_loss(
            yhat,
            dtrain,
            alpha=alpha,
            gamma=gamma):
        """
        Calculate gradient and hessian for Focal Loss,

        Args:
            yhat (np.array): Margin predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter

        Returns:
            grad: Focal Loss gradient
            hess: Focal Loss Hessian
        """

        grad = _gradient(yhat, dtrain, alpha=alpha, gamma=gamma)

        hess = _hessian(yhat, dtrain, alpha=alpha, gamma=gamma)

        return grad, hess

    return focal_loss
