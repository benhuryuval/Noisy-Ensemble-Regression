import numpy as np
from sklearn.tree import DecisionTreeRegressor

class LeastSquaresError:
    """Loss function for least squares (LS) estimation.
    Terminal regions do not need to be updated for least squares.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def __call__(y, raw_predictions):
        """Compute the least squares loss.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        return np.mean((y.ravel() - raw_predictions.ravel()) ** 2)

    def negative_gradient(y, raw_predictions):
        """Compute half of the negative gradient.
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.
        raw_predictions : ndarray of shape (n_samples,)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        return y.ravel() - raw_predictions.ravel()


from RobustGBR import LeastSquaresError
from sklearn.ensemble._gradient_boosting import predict_stages

class RobustGradientBoostingRegressor:
    def __init__(self,
                 loss="squared_error",
                 n_estimators=100,
                 subsample=1.0,
                 criterion='mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 noise_covariance=None,
                 learning_rate=0.1
                 ):

        self.estimators_ = np.empty((n_estimators, 1), dtype=object)
        self.train_score_ = np.zeros((n_estimators,), dtype=np.float64)
        self.loss_ = LeastSquaresError
        self.n_estimators_ = n_estimators

        self.loss=loss
        self.learning_rate=learning_rate
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.min_weight_fraction_leaf=min_weight_fraction_leaf
        self.max_depth=max_depth
        self.init=init
        self.subsample=subsample
        self.max_features=max_features
        self.min_impurity_decrease=min_impurity_decrease
        self.random_state=random_state
        self.alpha=alpha
        self.verbose=verbose
        self.max_leaf_nodes=max_leaf_nodes
        self.warm_start=warm_start
        self.validation_fraction=validation_fraction
        self.n_iter_no_change=n_iter_no_change
        self.tol=tol
        self.ccp_alpha=ccp_alpha
        self.noise_covariance=np.eye(n_estimators) if noise_covariance is None else noise_covariance

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = np.zeros(shape=(X.shape[0],1), dtype=np.float64)
        predict_stages(self.estimators_[:self.n_estimators_], X, self.learning_rate, raw_predictions)
        return raw_predictions


    def fit(self, X, y):
        # fit initial model and initialize raw predictions
        raw_predictions = np.zeros(
            shape=(X.shape[0], 1), dtype=np.float64
        )

        # fit the boosting stages
        begin_at_stage = 0

        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # track deviance (= loss)
            self.train_score_[i] = self.loss_.__call__(y, raw_predictions)

            # fit next stage of trees
            """Fit another stage of ``_n_classes`` trees to the boosting model."""
            residual = self.loss_.negative_gradient(
                y, raw_predictions
            )

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state,
                ccp_alpha=self.ccp_alpha,
            )
            tree.fit(X, residual, check_input=False)

            # add tree to ensemble
            self.estimators_[i, 0] = tree
            self.n_estimators_ = i + 1

            # update raw_predictions
            raw_predictions = self._raw_predict(X)

        return self

    def predict(self, X):
        return self._raw_predict(X)
