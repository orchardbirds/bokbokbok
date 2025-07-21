import numpy as np

from typing import Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import xgboost as xgb

def LogCoshMetric(XGBoost: bool = False) -> Callable:
    """
    Calculates the [Log Cosh Error](https://openreview.net/pdf?id=rkglvsC9Ym) as an alternative to
    Mean Absolute Error.
    Args:
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """
    def log_cosh_error(
        yhat: np.ndarray, 
        dtrain: "xgb.DMatrix", 
        XGBoost: bool = XGBoost
        ) -> Union[tuple[str, float], tuple[str, float, bool]]:
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
            return "LogCosh", float(np.sum(elements) / len(y))
        else:
            return "LogCosh", float(np.sum(elements) / len(y)), False

    return log_cosh_error


def RMSPEMetric(XGBoost: bool = False) -> Callable:
    """
    Calculates the Root Mean Squared Percentage Error:
    https://www.kaggle.com/c/optiver-realized-volatility-prediction/overview/evaluation

    The corresponding Loss function is Squared Percentage Error.
    Args:
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """
    def RMSPE(
        yhat: np.ndarray, 
        dtrain: "xgb.DMatrix", 
        XGBoost: bool = XGBoost) -> Union[tuple[str, float], tuple[str, float, bool]]:
        """
        Root Mean Squared Log Error.
        All input labels are required to be greater than -1.

        yhat: Predictions
        dtrain: The XGBoost / LightGBM dataset
        XGBoost (Bool): If XGBoost is to be implemented
        """

        y = dtrain.get_label()
        elements = ((y - yhat) / y) ** 2
        if XGBoost:
            return "RMSPE", float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return "RMSPE", float(np.sqrt(np.sum(elements) / len(y))), False

    return RMSPE