import numpy as np
from bokbokbok.utils import clip_sigmoid


def WeightedCrossEntropyMetric(alpha=0.5, XGBoost=False):
    """
    Calculates the Weighted Cross Entropy Metric by applying a weighting factor alpha.

    Args:
        alpha (float): The scale to be applied.
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """


    def weighted_cross_entropy_metric(yhat, dtrain, alpha=alpha, XGBoost=XGBoost):
        """
        Weighted Cross Entropy Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)
        elements = - alpha * y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
        if XGBoost:
            return f'WCE_alpha{alpha}', (np.sum(elements) / len(y))
        else:
            return f'WCE_alpha{alpha}', (np.sum(elements) / len(y)), False

    return weighted_cross_entropy_metric


def FocalMetric(alpha=1.0, gamma=2.0, XGBoost=False):
    """
    Implements alpha-weighted Focal Loss taken from https://arxiv.org/pdf/1708.02002.pdf

    Args:
        alpha (float): The scale to be applied.
        gamma (float): The focusing parameter to be applied
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function
    """

    def focal_metric(yhat, dtrain, alpha=alpha, gamma=gamma, XGBoost=XGBoost):
        """
        Weighted Focal Loss Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)

        elements = (- alpha * y * np.log(yhat) * np.power(1 - yhat, gamma) -
                    (1 - y) * np.log(1 - yhat) *  np.power(yhat, gamma))

        if XGBoost:
            return f'Focal_alpha{alpha}_gamma{gamma}', (np.sum(elements) / len(y))
        else:
            return f'Focal_alpha{alpha}_gamma{gamma}', (np.sum(elements)/ len(y)), False

    return focal_metric
