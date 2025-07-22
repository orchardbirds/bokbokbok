# Welcome to the bokbokbok doks!

<img src="img/bokbokbok.png" width=190 align="right">

**bokbokbok** is a Python library that lets us easily implement custom loss functions and eval metrics in LightGBM and XGBoost.

## Example Usage - Weighted Cross Entropy

```python
params = {"objective": WeightedCrossEntropyLoss(alpha=alpha)}
clf = lgb.train(params=params,
                train_set=train,
                valid_sets=[train, valid],
                valid_names=['train','valid'],
                feval=WeightedCrossEntropyMetric(alpha=alpha))
```
## Licence
bokbokbok is created under the MIT License, see more in the LICENSE file

