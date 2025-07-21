from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from bokbokbok.loss_functions.classification import WeightedFocalLoss, WeightedCrossEntropyLoss
from bokbokbok.eval_metrics.classification import WeightedFocalMetric, WeightedCrossEntropyMetric
from bokbokbok.utils import clip_sigmoid
import lightgbm as lgb


def test_focal_lgb_implementation():
    """
    Assert that there is no difference between running Focal with alpha=1 and gamma=0
    and LightGBM's internal CE loss.
    """
    X, y = make_classification(n_samples=1000, 
                               n_features=10, 
                               random_state=41114)

    X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                          y, 
                                                          test_size=0.25, 
                                                          random_state=41114)

    alpha = 1.0
    gamma = 0

    train = lgb.Dataset(X_train, y_train)
    valid = lgb.Dataset(X_valid, y_valid, reference=train)

    params_wfl = {
     "n_estimators": 300,
     "seed": 41114,
     "n_jobs": 8,
     "learning_rate": 0.1,
     "early_stopping_rounds": 100,
     "objective": WeightedFocalLoss(alpha=alpha, gamma=gamma)
   }

    wfl_clf = lgb.train(params=params_wfl,
                train_set=train,
                valid_sets=[train, valid],
                valid_names=["train", "valid"],
                feval=WeightedFocalMetric(alpha=alpha, gamma=gamma)
                )


    params = {
        "n_estimators": 300,
        "objective": "cross_entropy",
        "seed": 41114,
        "n_jobs": 8,
        "metric": "cross_entropy",
        "learning_rate": 0.1,
        "boost_from_average": False,
        "early_stopping_rounds": 100,
    }

    clf = lgb.train(params=params,
                    train_set=train,
                    valid_sets=[train, valid],
                    valid_names=["train", "valid"])

    wfl_preds = clip_sigmoid(wfl_clf.predict(X_valid))
    preds = clf.predict(X_valid)
    assert mean_absolute_error(wfl_preds, preds) == 0.0


def test_focal_wce_comparison():
    """
    Assert that there is no difference between running Focal with alpha=3 and gamma=0
    and running WCE with alpha=3.
    """
    X, y = make_classification(n_samples=1000, 
                               n_features=10, 
                               random_state=41114)

    X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                          y, 
                                                          test_size=0.25, 
                                                          random_state=41114)

    alpha = 3.0
    gamma = 0

    train = lgb.Dataset(X_train, y_train)
    valid = lgb.Dataset(X_valid, y_valid, reference=train)

    params_wfl = {
     "n_estimators": 300,
     "seed": 41114,
     "n_jobs": 8,
     "learning_rate": 0.1,
     "early_stopping_rounds": 100,
     "objective": WeightedFocalLoss(alpha=alpha, gamma=gamma)
   }

    wfl_clf = lgb.train(params=params_wfl,
                train_set=train,
                valid_sets=[train, valid],
                valid_names=['train','valid'],
                feval=WeightedFocalMetric(alpha=alpha, gamma=gamma))


    params_wce = {
     "n_estimators": 300,
     "seed": 41114,
     "n_jobs": 8,
     "learning_rate": 0.1,
     "early_stopping_rounds": 100,
     "objective": WeightedCrossEntropyLoss(alpha=alpha)
   }

    wce_clf = lgb.train(params=params_wce,
                train_set=train,
                valid_sets=[train, valid],
                valid_names=["train", "valid"],
                feval=WeightedCrossEntropyMetric(alpha=alpha))

    wfl_preds = clip_sigmoid(wfl_clf.predict(X_valid))
    wce_preds = clip_sigmoid(wce_clf.predict(X_valid))
    assert mean_absolute_error(wfl_preds, wce_preds) == 0.0
