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
        X,
        y,
        max_depth: int = 4,
        min_sample_leaf: int = 2,
        NoiseCov: float = 0,
        RobustFlag = 1
    ):

        # Saving the train dataset
        self.X, self.y = X, y

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate and coefficients (weights)
        self.gamma = []  # np.array([1])

        # Saving the noise covariance matrix
        self.NoiseCov = NoiseCov

        # Saving robust learner indicator
        self.RobustFlag = RobustFlag

        # Weak learner list
        self.weak_learners = []

        # Setting the current iteration m to 1
        self.cur_m = 0

        # Saving the y_mean as the most recent prediction
        n_samples = len(y)
        self.y_mean = np.mean(y)  # mean target value over dataset
        self._predictions = self.y_mean * np.ones(n_samples).reshape(n_samples, 1)  # initialize regressor to mean target value

        # Initialize residuals
        self._residuals = y - self._predictions

        # Saving previous predictions of weak-learners (for coefficient calculation)
        self._predictions_matrix = self._predictions

    def fit(self, X, y, m: int = 10):
        """
        Applies the iterative algorithm
        """

        # Iterating over the number of estimators
        for _ in range(self.cur_m+1, self.cur_m+m+1):
            # Growing the tree on the residuals
            _weak_learner = Tree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_sample_leaf
            )
            _weak_learner.fit(X, self._residuals)
            self.weak_learners.append(_weak_learner)  # Appending the weak learner to the list

            # Getting the weak learner predictions
            _predictions_wl = _weak_learner.predict(X).reshape(len(y), 1)

            # Setting the weak learner weight
            y_minus_f = np.subtract(y, self._predictions)
            sum_phi_y_minus_f = np.mean(np.multiply(_predictions_wl, y_minus_f))
            if _ == 1:  # first weak-learner (after initialization)
                sum_gamma_sigma = np.array([0])
            else:
                gamma = self.gamma.reshape(_-1, 1)
                sum_gamma_sigma = gamma.T.dot(self.NoiseCov[0:_-1, _])
            phi_sqrd = np.mean(_predictions_wl**2)
            new_gamma = (sum_phi_y_minus_f + self.RobustFlag * sum_gamma_sigma) / (phi_sqrd + self.RobustFlag * self.NoiseCov[_, _])

            # Adding new weight to list
            if _ == 1:
                self.gamma = new_gamma
            else:
                self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)

            # Saving the current predictions
            self._predictions = self._predictions + new_gamma * _predictions_wl

            # Updating the residuals
            self._residuals = np.subtract(y, self._predictions)

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, X):
        """
        Given the dictionary, predict the value of the y variable
        """
        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), self.NoiseCov, X.shape[0])

        # Starting from the (noisy) mean
        yhat = self.y_mean + pred_noise[:, 0]

        # And aggregating (noisy) predictions
        for _m in range(self.cur_m):
            noisy_pred = self.weak_learners[_m].predict(X) + pred_noise[:, _m+1]
            yhat += self.gamma[_m] * noisy_pred
        return yhat
