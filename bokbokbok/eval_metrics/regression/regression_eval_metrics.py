import numpy as np


def SquaredLogErrorMetric():
    def squared_log_error(yhat, dtrain):
        """
        Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        """

        y = dtrain.get_label()
        yhat[yhat < -1] = -1 + 1e-6
        elements = 0.5 * np.power(np.log1p(y) - np.log1p(yhat), 2)
        return 'SLE', float(np.sum(elements) / len(y)), False

    return squared_log_error


def RootMeanSquaredLogErrorMetric():
    def root_mean_squared_log_error(yhat, dtrain):
        """
        Root Mean Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        """

        y = dtrain.get_label()
        yhat[yhat < -1] = -1 + 1e-6
        elements = np.power(np.log1p(y) - np.log1p(yhat), 2)
        return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y))), False

    return root_mean_squared_log_error
