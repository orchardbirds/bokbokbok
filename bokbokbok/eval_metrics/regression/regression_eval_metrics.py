import numpy as np


def LogCoshMetric(XGBoost=False):
    """
    Calculates the [Log Cosh Error](https://openreview.net/pdf?id=rkglvsC9Ym) as an alternative to
    Mean Absolute Error.
    Args:
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """
    def log_cosh_error(yhat, dtrain, XGBoost=XGBoost):
        """
        Root Mean Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        XGBoost (Bool): If XGBoost is to be implemented
        """

        y = dtrain.get_label()
        elements = np.log(np.cosh(yhat - y))
        if XGBoost:
            return 'LogCosh', float(np.sum(elements) / len(y))
        else:
            return 'LogCosh', float(np.sum(elements) / len(y)), False

    return log_cosh_error
