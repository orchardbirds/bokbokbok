import numpy as np


def SquaredLogErrorMetric(XGBoost=False):
    """
    Calculates the Squared Log Error shown here:
    https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html

    Args:
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """
    def squared_log_error(yhat, dtrain, XGBoost=XGBoost):
        """
        Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        XGBoost (Bool): If XGBoost is to be implemented
        """

        y = dtrain.get_label()
        yhat[yhat < -1] = -1 + 1e-15
        elements = 0.5 * np.power(np.log1p(y) - np.log1p(yhat), 2)
        if XGBoost:
            return 'SLE', float(np.sum(elements) / len(y))
        else:
            return 'SLE', float(np.sum(elements) / len(y)), False

    return squared_log_error


def RootMeanSquaredLogErrorMetric(XGBoost=False):
    """
    Calculates the Root Mean Squared Log Error shown here:
    https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html

    Args:
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """
    def root_mean_squared_log_error(yhat, dtrain, XGBoost=XGBoost):
        """
        Root Mean Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        XGBoost (Bool): If XGBoost is to be implemented
        """

        y = dtrain.get_label()
        yhat[yhat < -1] = -1 + 1e-15
        elements = np.power(np.log1p(y) - np.log1p(yhat), 2)
        if XGBoost:
            return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y))), False

    return root_mean_squared_log_error


def LogCoshMetric(XGBoost=False):
    """
    Calculates the Log Cosh Error
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
