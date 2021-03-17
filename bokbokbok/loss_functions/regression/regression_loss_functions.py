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


def LogCoshLoss():
    """
    An alternative to Mean Absolute Error.
    """

    def _gradient(yhat, dtrain):
        """Compute the log cosh gradient.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            log cosh gradient
        """

        y = dtrain.get_label()
        return np.tanh(yhat - y)

    def _hessian(yhat, dtrain):
        """Compute the log cosh hessian.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            log cosh Hessian
        """

        y = dtrain.get_label()
        return 1. / np.power(np.cosh(yhat - y), 2)

    def log_cosh_loss(
            yhat,
            dtrain
    ):
        """
        Calculate gradient and hessian for log cosh loss.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            grad: log cosh loss gradient
            hess: log cosh loss Hessian
        """
        grad = _gradient(yhat, dtrain)

        hess = _hessian(yhat, dtrain)

        return grad, hess

    return log_cosh_loss
