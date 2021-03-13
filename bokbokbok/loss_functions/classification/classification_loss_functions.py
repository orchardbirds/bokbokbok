import numpy as np


def WeightedCrossEntropyLoss(alpha=0.5):
    """
    Calculates the Weighted Cross-Entropy Loss, which applies
    a factor alpha to the regular Cross-Entropy Loss.
    """
    if float(alpha) == 1.0:
        raise UserWarning('Using alpha == 1, it is better to use the already existing Cross Entropy Metric')

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

        yhat = 1. / (1. + np.exp(-yhat))
        yhat[yhat >= 1] = 1 - 1e-6
        yhat[yhat <= 0] = 1e-6

        grad = yhat - (y * alpha)

        return grad

    def _hessian(yhat):
        """Compute the weighted cross-entropy hessian.

        Args:
            yhat (np.array): Margin predictions

        Returns:
            hess: Weighted cross-entropy Hessian
        """

        yhat = 1. / (1. + np.exp(-yhat))
        yhat[yhat >= 1] = 1 - 1e-6
        yhat[yhat <= 0] = 1e-6

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

