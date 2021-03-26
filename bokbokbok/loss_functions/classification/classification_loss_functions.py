import numpy as np
from bokbokbok.utils import clip_sigmoid


def WeightedCrossEntropyLoss(alpha=0.5):
    """
    Calculates the Weighted Cross-Entropy Loss, which applies a factor alpha, allowing one to
    trade off recall and precision by up- or down-weighting the cost of a positive error relative
    to a negative error.

    A value alpha > 1 decreases the false negative count, hence increasing the recall.
    Conversely, setting alpha < 1 decreases the false positive count and increases the precision. 
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

        grad = (y * yhat * (alpha - 1)) + yhat - (alpha * y)

        return grad

    def _hessian(yhat, dtrain, alpha):
        """Compute the weighted cross-entropy hessian.

        Args:
            yhat (np.array): Margin predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            hess: Weighted cross-entropy Hessian
        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)

        hess = (y * (alpha - 1) + 1) * yhat * (1 - yhat)

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

        hess = _hessian(yhat, dtrain, alpha=alpha)

        return grad, hess

    return weighted_cross_entropy


def WeightedFocalLoss(alpha=1.0, gamma=2.0):
    """
    Calculates the [Weighted Focal Loss.](https://arxiv.org/pdf/1708.02002.pdf)

    Note that if using alpha = 1 and gamma = 0,
    this is the same as using regular Cross Entropy.

    The more gamma is increased, the more the model is focussed on the hard, misclassified examples.

    A value alpha > 1 decreases the false negative count, hence increasing the recall.
    Conversely, setting alpha < 1 decreases the false positive count and increases the precision. 

    """

    def _gradient(yhat, dtrain, alpha, gamma):
        """Compute the weighted focal gradient.

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

        grad = (
                alpha * y * np.power(1 - yhat, gamma) * (gamma * yhat * np.log(yhat) + yhat - 1) +
                (1 - y) * np.power(yhat, gamma) * (yhat - gamma * np.log(1 - yhat) * (1 - yhat))
                )

        return grad

    def _hessian(yhat, dtrain, alpha, gamma):
        """Compute the weighted focal hessian.

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

        hess = (
                alpha * y * yhat * np.power(1 - y,
                                            gamma) * (gamma * (1 - yhat) * np.log(yhat) + 2 * gamma * (1 - yhat) -
                                                      np.power(gamma, 2) * yhat * np.log(yhat) + 1 - yhat) +
                (1 - y) * np.power(yhat, gamma + 1) * (1 - yhat) * (2 * gamma + gamma * (np.log(1 - yhat)) + 1)
                )

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
