from sklearn.metrics import cohen_kappa_score
import numpy as np
from typing import Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import xgboost as xgb

def QuadraticWeightedKappaMetric(XGBoost: bool = False) -> Callable:
    """
    Calculates the Weighted Cross Entropy Metric by applying a weighting factor alpha, allowing one to
    trade off recall and precision by up- or down-weighting the cost of a positive error relative to a
    negative error.

    A value alpha > 1 decreases the false negative count, hence increasing the recall.
    Conversely, setting alpha < 1 decreases the false positive count and increases the precision. 

    Args:
        alpha (float): The scale to be applied.
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function

    """


    def quadratic_weighted_kappa_metric(
        yhat: np.ndarray,
        dtrain: "xgb.DMatrix", 
        XGBoost: bool = XGBoost) -> Union[tuple[str, float], tuple[str, float, bool]]:
        """
        Weighted Cross Entropy Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to maximise function

        """
        y = dtrain.get_label()
        num_class = len(np.unique(dtrain.get_label()))

        if not XGBoost:
            # LightGBM needs extra reshaping
            yhat = yhat.reshape(num_class, len(y)).T
        yhat = yhat.argmax(axis=1)

        qwk = cohen_kappa_score(y, yhat, weights="quadratic")

        if XGBoost:
            return "QWK", qwk
        else:
            return "QWK", qwk, True

    return quadratic_weighted_kappa_metric
