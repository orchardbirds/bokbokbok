import numpy as np
from bokbokbok.utils import clip_sigmoid


def WeightedCrossEntropyMetric(alpha=0.5):

    def weighted_cross_entropy_metric(yhat, dtrain, alpha=alpha):
        """
        Weighted Cross Entropy Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)
        elements = - alpha * y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
        return f'WCE_alpha{alpha}', (np.sum(elements) / len(y)), False

    return weighted_cross_entropy_metric


def FocalMetric(alpha=1.0, gamma=2.0):
    """
    Implements alpha-weighted Focal Loss taken from https://arxiv.org/pdf/1708.02002.pdf
    """

    def focal_metric(yhat, dtrain, alpha=alpha, gamma=gamma):
        """
        Weighted Focal Loss Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)

        elements = (- alpha * y * np.log(yhat) * np.power(1 - yhat, gamma) -
                    (1 - y) * np.log(1 - yhat) *  np.power(yhat, gamma))
        return f'Focal_alpha{alpha}_gamma{gamma}', (np.sum(elements)/ len(y)), False

    return focal_metric
