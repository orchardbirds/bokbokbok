import numpy as np


def WeightedCrossEntropyMetric(alpha=0.5):

    if alpha == 1.0:
        raise UserWarning('Using alpha == 1, it is better to use the already existing Cross Entropy Metric')

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
        yhat = 1. / (1. + np.exp(-yhat))
        yhat[yhat >= 1] = 1 - 1e-6
        yhat[yhat <= 0] = 1e-6
        elements = alpha * y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
        return 'WCE', (np.sum(elements) * -1 / len(y)), False

    return weighted_cross_entropy_metric
