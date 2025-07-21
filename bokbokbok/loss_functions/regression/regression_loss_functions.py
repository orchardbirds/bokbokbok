import numpy as np

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost as xgb

def LogCoshLoss() -> Callable:
    """
    [Log Cosh Loss](https://openreview.net/pdf?id=rkglvsC9Ym) is an alternative to Mean Absolute Error.
    """

    def _gradient(yhat: np.ndarray, dtrain: "xgb.DMatrix") -> np.ndarray:
        """Compute the log cosh gradient.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            log cosh gradient
        """

        y = dtrain.get_label()
        return -np.tanh(y - yhat)

    def _hessian(yhat: np.ndarray, dtrain: "xgb.DMatrix") -> np.ndarray:
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
            yhat: np.ndarray,
            dtrain: "xgb.DMatrix"
    ) -> tuple[np.ndarray, np.ndarray]:
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


def SPELoss() -> Callable:
    """
    Squared Percentage Error loss
    """

    def _gradient(yhat: np.ndarray, dtrain: "xgb.DMatrix") -> np.ndarray:
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

    def _hessian(dtrain: "xgb.DMatrix") -> np.ndarray:
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

    def squared_percentage(
        yhat: np.ndarray, 
        dtrain: "xgb.DMatrix"
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradient and hessian for squared percentage error.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset

        Returns:
            grad: SPE loss gradient
            hess: SPE loss Hessian
        """
        grad = _gradient(yhat, dtrain)

        hess = _hessian(dtrain)

        return grad, hess

    return squared_percentage