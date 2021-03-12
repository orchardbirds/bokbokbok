def WeightedCrossEntropyLoss(alpha=0.5):
    """
    Calculates the Weighted Cross-Entropy Loss, which applies
    a factor alpha to the regular Cross-Entropy Loss.
    """
    if alpha == 1.0:
        raise UserWarning('Using alpha == 1, it is better to use the already existing Cross Entropy Metric')

    def _gradient(yhat, dtrain, alpha):
        """Compute the weighted cross-entropy gradient.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            grad: Weighted cross-entropy gradient
        """
        y = dtrain.get_label()

        grad = (-alpha * y / yhat) + ((1 - y) / (1 - yhat))

        return grad

    def _hessian(yhat, dtrain, alpha):
        """Compute the weighted cross-entropy hessian.

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            hess: Weighted cross-entropy Hessian
        """

        y = dtrain.get_label()

        hess = (alpha * y / yhat ** 2) + ((1 - y) / (1 - yhat) ** 2)

        return hess

    def weighted_cross_entropy(
            yhat,
            dtrain,
            alpha=alpha
    ):
        """
        Calculate gradient and hessian for weight cross-entropy,

        Args:
            yhat (np.array): Predictions
            dtrain: The XGBoost / LightGBM dataset
            alpha (float): Scale applied

        Returns:
            grad: Weighted cross-entropy gradient
            hess: Weighted cross-entropy Hessian
        """
        yhat[yhat >= 1] = 1 - 1e-6
        yhat[yhat <= 0] = 1e-6
        grad = _gradient(yhat, dtrain, alpha=alpha)

        hess = _hessian(yhat, dtrain, alpha=alpha)

        return grad, hess

    return weighted_cross_entropy

