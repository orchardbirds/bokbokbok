import numpy as np


def LogCoshLoss():
    """
    [Log Cosh Loss](https://openreview.net/pdf?id=rkglvsC9Ym) is an alternative to Mean Absolute Error.
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
        return -np.tanh(y - yhat)

    def _hessian(yhat, dtrain):
        """Compute the log cosh hessian.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            log cosh Hessian
        """

        y = dtrain.get_label()
        return 1. / np.power(np.cosh(y - yhat), 2)

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


def SPELoss():
    """
    Squared Percentage Error loss
    """

    def _gradient(yhat, dtrain):
        """
        Compute the gradient squared percentage error.
        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            SPE Gradient
        """
        y = dtrain.get_label()
        return -2*(y-yhat)/(y**2)

    def _hessian(yhat, dtrain):
        """
        Compute the hessian for squared percentage error.
        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            SPE Hessian
        """
        y = dtrain.get_label()
        return 2/(y**2)

    def squared_percentage(yhat, dtrain):
        """
        Calculate gradient and hessian for squared percentage error.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            grad: SPE loss gradient
            hess: SPE loss Hessian
        """
        #yhat[yhat < -1] = -1 + 1e-6
        grad = _gradient(yhat, dtrain)

        hess = _hessian(yhat, dtrain)

        return grad, hess

    return squared_percentage