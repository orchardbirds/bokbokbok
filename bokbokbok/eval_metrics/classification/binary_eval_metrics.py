import numpy as np
from sklearn.metrics import f1_score
from bokbokbok.utils import clip_sigmoid
from typing import Any, Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import xgboost as xgb
def WeightedCrossEntropyMetric(
    alpha: float = 0.5, 
    XGBoost: bool = False
    ) -> Callable:
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


    def weighted_cross_entropy_metric(
        yhat: np.ndarray, 
        dtrain: "xgb.DMatrix", 
        alpha=alpha, 
        XGBoost=XGBoost
        ) -> Union[tuple[str, float], tuple[str, float, bool]]:
        """
        Weighted Cross Entropy Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)
        elements = - alpha * y * np.log(yhat) - (1 - y) * np.log(1 - yhat)
        if XGBoost:
            return f"WCE_alpha{alpha}", (np.sum(elements) / len(y))
        else:
            return f"WCE_alpha{alpha}", (np.sum(elements) / len(y)), False

    return weighted_cross_entropy_metric


def WeightedFocalMetric(
    alpha: float = 1.0, 
    gamma: float = 2.0, 
    XGBoost: bool = False
    ) -> Callable:
    """
    Implements [alpha-weighted Focal Loss](https://arxiv.org/pdf/1708.02002.pdf)

    The more gamma is increased, the more the model is focussed on the hard, misclassified examples.

    A value alpha > 1 decreases the false negative count, hence increasing the recall.
    Conversely, setting alpha < 1 decreases the false positive count and increases the precision. 

    Args:
        alpha (float): The scale to be applied.
        gamma (float): The focusing parameter to be applied
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=False` in the XGBoost train function
    """

    def focal_metric(
        yhat: np.ndarray, 
        dtrain: "xgb.DMatrix", 
        alpha: float = alpha, 
        gamma: float = gamma, 
        XGBoost: bool = XGBoost) -> Union[tuple[str, float], tuple[str, float, bool]]:
        """
        Weighted Focal Loss Metric.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied
            gamma (float): Focusing parameter
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to minimise function

        """
        y = dtrain.get_label()
        yhat = clip_sigmoid(yhat)

        elements = (- alpha * y * np.log(yhat) * np.power(1 - yhat, gamma) -
                    (1 - y) * np.log(1 - yhat) * np.power(yhat, gamma))

        if XGBoost:
            return f'Focal_alpha{alpha}_gamma{gamma}', (np.sum(elements) / len(y))
        else:
            return f'Focal_alpha{alpha}_gamma{gamma}', (np.sum(elements) / len(y)), False

    return focal_metric


def F1_Score_Binary(
    XGBoost: bool = False,
    *args: Any, 
    **kwargs: Any,
    ) -> Callable:
    """
    Implements the f1_score metric
    [from scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn-metrics-f1-score)

    Args:
        *args: The arguments to be fed into the scikit learn metric.
        XGBoost (Bool): Set to True if using XGBoost. We assume LightGBM as default use.
                        Note that you should also set `maximize=True` in the XGBoost train function

    """
    def binary_f1_score(
        yhat: np.ndarray, 
        data: "xgb.DMatrix", 
        XGBoost: bool = XGBoost
        ) -> Union[tuple[str, Any], tuple[str, Any, bool]]: # needs better typing for f1 but I don't care
        """
        F1 Score.

        Args:
            yhat: Predictions
            dtrain: The XGBoost / LightGBM dataset
            XGBoost (Bool): If XGBoost is to be implemented

        Returns:
            Name of the eval metric, Eval score, Bool to maximise function
        """
        y_true = data.get_label()
        yhat = np.round(yhat)
        if XGBoost:
            return "F1", f1_score(y_true, yhat, *args, **kwargs)
        else:
            return "F1", f1_score(y_true, yhat, *args, **kwargs), True

    return binary_f1_score
