# The base class with the weak learner 
# from GradientBoosting.tree import Tree
from sklearn.tree import DecisionTreeRegressor as Tree

# Data wrangling 
import pandas as pd
import numpy as np

# Directory traversal 
import os 
import shutil

# Python infinity
from math import inf

# Ploting 
import matplotlib.pyplot as plt

class RobustRegressionGB():
    """
    Class that implements the regression gradient boosting algorithm
    """
    def __init__(
        self,
        d: pd.DataFrame,
        y_var: str,
        x_vars: list,
        max_depth: int = 4,
        min_sample_leaf: int = 2,
        learning_rate: float = 0.4,
        NoiseCov: float = 0
    ):
        # Saving the names of y variable and X features
        self.y_var = y_var
        self.x_vars = x_vars

        # Saving the node data to memory
        self.d = d[[y_var] + x_vars].copy()

        # Saving the number of observations in data
        self.n = len(d)

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate and coefficients (weights)
        self.learning_rate = learning_rate
        self.gamma = np.array([1])

        # Saving the noise covariance matrix
        self.NoiseCov = NoiseCov

        # Weak learner list
        self.weak_learners = []

        # Setting the current iteration m to 1
        self.cur_m = 0

        # Saving the y_mean as the most recent prediction
        self.y_mean = np.mean(self.d[y_var].values)  # mean target value over dataset
        self._predictions = self.y_mean * np.ones(self.n).reshape(self.n, 1)  # initialize regressor to mean target value
        # self.mean_predictions = self.y_mean.reshape(1)

        # Initialize residuals
        self._residuals = self.d[self.y_var].values.reshape(self.n, 1) - self._predictions

        # Saving previous predictions of weak-learners (for coefficient calculation)
        self._predictions_matrix = self._predictions

        # Setting prediction covariance matrix
        # self.PredictionCov = [[np.var(self._predictions[0, :])]]

    def fit(self, m: int = 10):
        """
        Applies the iterative algorithm
        """
        # Converting the X to suitable inputs
        _inputs = self.d[self.x_vars].values.reshape(self.n, 1)
        _y = self.d[self.y_var].values.reshape(self.n, 1)

        # Iterating over the number of estimators
        for _ in range(self.cur_m+1, self.cur_m + m+1):
            # Growing the tree on the residuals
            _weak_learner = Tree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_sample_leaf
            )
            _weak_learner.fit(_inputs, self._residuals)
            self.weak_learners.append(_weak_learner)  # Appending the weak learner to the list

            # Getting the weak learner predictions
            _predictions_wl = _weak_learner.predict(_inputs).reshape(self.n, 1)

            # Update prediction covariance matrix
            # b, A = _predictions_wl, self._predictions_matrix
            #
            # new_row = np.dot(A-A.mean(axis=1).reshape(_, 1), b.T-b.mean(axis=1)) / (b.shape[1]-1)
            # self.PredictionCov = np.concatenate((self.PredictionCov, new_row), axis=0)
            # new_col = np.concatenate((new_row[0, :], [np.var(b)])).reshape(-1, 1)
            # self.PredictionCov = np.concatenate((self.PredictionCov, new_col), axis=1)

            # Setting the weak learner weight
            # self.mean_predictions = np.concatenate((self.mean_predictions, np.mean(_predictions_wl).reshape(1)), axis=0)
            # self.gamma.append(self.y_mean * np.linalg.inv(self.NoiseCov[0:_+1,0:_+1]+
            phi = _predictions_wl
            y_minus_f = np.subtract(_y, self._predictions)
            phi_sqrd = np.mean(_predictions_wl**2)
            gamma = self.gamma.reshape(np.sum(self.gamma.shape), 1)
            new_gamma = (np.mean(np.multiply(phi, y_minus_f)) + gamma.T.dot(self.NoiseCov[0:_, _])) / (phi_sqrd + self.NoiseCov[_, _])

            self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)

            # Saving the current predictions
            # self._predictions = [self._predictions[i] + self.gamma[_] * _predictions_wl[i] for i in range(self.n)]
            # self._predictions = self.gamma[0:_] * self._predictions_matrix + self.gamma[_][-1] * _predictions_wl
            # self._predictions_matrix = np.concatenate((self._predictions_matrix, self._predictions), axis=0)
            self._predictions = self._predictions + new_gamma * _predictions_wl

            # Updating the residuals
            self._residuals = _y - self._predictions

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, x):
        """
        Given the dictionary, predict the value of the y variable
        """
        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), self.NoiseCov, len(x))

        # Starting from the mean
        # yhat = self._predictions + pred_noise[:, 0]
        yhat = self.y_mean + pred_noise[:, 0]

        # And aggregating predictions
        for _m in range(self.cur_m):
            noisy_pred = self.weak_learners[_m].predict(x.reshape(len(x), 1)) + pred_noise[0, _m+1]
            yhat += self.gamma[_m+1] * noisy_pred
        return yhat
