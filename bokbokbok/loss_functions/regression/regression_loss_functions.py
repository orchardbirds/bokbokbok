import numpy as np


def SquaredLogErrorLoss():
    """
    All input labels are required to be greater than -1.
    """

    def _gradient(yhat, dtrain):
        """Compute the squared log error gradient.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            squared log error gradient
        """

        y = dtrain.get_label()
        return (np.log1p(yhat) - np.log1p(y)) / (yhat + 1)

    def _hessian(yhat, dtrain):
        """Compute the squared log hessian.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            squared log Hessian
        """

        y = dtrain.get_label()
        return ((-np.log1p(yhat) + np.log1p(y) + 1) /
                np.power(yhat + 1, 2))

    def squared_log_loss(
            yhat,
            dtrain
    ):
        """
        Calculate gradient and hessian for squared log loss.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            grad: squared log loss gradient
            hess: squared log loss Hessian
        """
        yhat[yhat < -1] = -1 + 1e-6
        grad = _gradient(yhat, dtrain)

        hess = _hessian(yhat, dtrain)

        return grad, hess

    return squared_log_loss