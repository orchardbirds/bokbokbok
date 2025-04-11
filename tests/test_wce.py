from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from bokbokbok.loss_functions.classification import WeightedCrossEntropyLoss
from bokbokbok.eval_metrics.classification import WeightedCrossEntropyMetric
from bokbokbok.utils import clip_sigmoid
import lightgbm as lgb
import xgboost as xgb


def test_wce_lgb_implementation():
    """
    Assert that there is no difference between running WCE with alpha=1
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

    train = lgb.Dataset(X_train, y_train)
    valid = lgb.Dataset(X_valid, y_valid, reference=train)

    params_wce = {
     "n_estimators": 300,
     "seed": 41114,
     "n_jobs": 8,
     "learning_rate": 0.1,
     "objective": WeightedCrossEntropyLoss(alpha=alpha),
     "early_stopping_rounds": 100
   }

    wce_clf = lgb.train(params=params_wce,
                train_set=train,
                valid_sets=[train, valid],
                valid_names=["train","valid"],
                feval=WeightedCrossEntropyMetric(alpha=alpha),
                )


    params = {
        "n_estimators": 300,
        "objective": "cross_entropy",
        "seed": 41114,
        "n_jobs": 8,
        "metric": "cross_entropy",
        "learning_rate": 0.1,
        "boost_from_average": False,
        "early_stopping_rounds": 100
    }

    clf = lgb.train(params=params,
                    train_set=train,
                    valid_sets=[train, valid],
                    valid_names=["train","valid"],
                    )

    wce_preds = clip_sigmoid(wce_clf.predict(X_valid))
    preds = clf.predict(X_valid)
    assert mean_absolute_error(wce_preds, preds) == 0.0


def test_wce_xgb_implementation():
    """
    Assert that there is no difference between running WCE with alpha=1
    and XGBoost's internal CE loss.
    """
    X, y = make_classification(n_samples=1000, 
                               n_features=10, 
                               random_state=41114)

    X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                          y, 
                                                          test_size=0.25, 
                                                          random_state=41114)

    alpha = 1

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    params_wce = {
        "seed": 41114,
        "learning_rate": 0.1,
        "disable_default_eval_metric": True
    }

    results = {}
    bst_wce = xgb.train(params_wce,
            dtrain=dtrain,
            num_boost_round=300,
            early_stopping_rounds=10,
            verbose_eval=1,
            obj=WeightedCrossEntropyLoss(alpha=alpha),
            maximize=False,
            custom_metric=WeightedCrossEntropyMetric(alpha=alpha, XGBoost=True),
            evals=[(dtrain, "dtrain"), (dvalid, "dvalid")],
            evals_result=results)


    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    params = {
        "seed": 41114,
        "objective": "reg:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.1,
    }
    results = {}
    bst = xgb.train(params,
            dtrain=dtrain,
            num_boost_round=300,
            early_stopping_rounds=10,
            verbose_eval=1,
            evals=[(dtrain, "dtrain"), (dvalid, "dvalid")],
            evals_result=results)

    wce_preds = clip_sigmoid(bst_wce.predict(dvalid, iteration_range = (0, bst_wce.best_iteration)))
    preds = bst.predict(dvalid, iteration_range = (0, bst.best_iteration))
    print(preds)
    print(wce_preds)
    assert mean_absolute_error(wce_preds, preds) == 0.0