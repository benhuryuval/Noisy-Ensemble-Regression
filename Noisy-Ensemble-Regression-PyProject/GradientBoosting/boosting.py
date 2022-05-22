# The base class with the weak learner 
from GradientBoosting.tree import Tree

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


class RegressionGB():
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
    ):
        # Saving the names of y variable and X features
        self.y_var = y_var 
        self.features = x_vars

        # Saving the node data to memory 
        self.d = d[[y_var] + x_vars].copy()

        # Saving the data to the node 
        self.Y = d[y_var].values.tolist()

        # Saving the number of observations in data
        self.n = len(d)

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate 
        self.learning_rate = learning_rate

        # Weak learner list 
        self.weak_learners = []

        # Setting the current iteration m to 0
        self.cur_m = 0

        # Saving the mean of y
        self.y_mean = self.get_mean(self.Y)

        # Saving the y_mean as the most recent prediction 
        self._predictions = [self.y_mean] * self.n

    @staticmethod
    def get_mean(x: list) -> float:
        """
        Calculates the mean over a list of float elements
        """
        # Initiating the sum counter 
        _sum = 0 

        # Infering the lenght of list 
        _n = len(x)

        if _n == 0:
            return inf

        # Iterating through the y values
        for _x in x:
            _sum += _x

        # Returning the mean 
        return _sum / _n

    def fit(
        self, 
        m: int = 10
        ):
        """
        Applies the iterative algorithm 
        """
        # Converting the X to suitable inputs
        _inputs = self.d[self.features].to_dict('records')

        # Saving the gamma list to memory 
        self.gamma = []

        # Iterating over the number of estimators
        for _ in range(self.cur_m, self.cur_m + m):
            # Calculating the residuals
            _residuals = [self.Y[i] - self._predictions[i] for i in range(self.n)]

            # Saving the current iterations residuals to the original dataframe 
            _r_name = f"residuals"
            self.d[_r_name] = _residuals

            # Creating a weak learner 
            _weak_learner = Tree(
                d = self.d.copy(), 
                y_var = _r_name,
                x_vars = self.features,
                max_depth = self.max_depth,
                min_sample_leaf = self.min_sample_leaf,
            )

            # Growing the tree on the residuals
            _weak_learner.fit()

            # Appending the weak learner to the list
            self.weak_learners.append(_weak_learner)

            # Getting the weak learner predictions
            _predictions_wl = [_weak_learner.predict(_x) for _x in _inputs] 

            # Updating the current predictions
            self._predictions = [self._predictions[i] + self.learning_rate * _predictions_wl[i] for i in range(self.n)]

        # Incrementing the current iteration 
        self.cur_m += m

    def predict(self, x: dict) -> float:
        """
        Given the dictionary, predict the value of the y variable
        """
        # Starting from the mean
        yhat = self.y_mean

        for _m in range(self.cur_m):
            yhat += self.learning_rate * self.weak_learners[_m].predict(x)

        return yhat

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
        self.features = x_vars

        # Saving the node data to memory
        self.d = d[[y_var] + x_vars].copy()

        # Saving the data to the node
        self.Y = d[y_var].values.tolist()

        # Saving the number of observations in data
        self.n = len(d)

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate and coefficients (weights)
        self.learning_rate = learning_rate
        self.gamma = [1]

        # Saving the noise covariance matrix
        self.NoiseCov = NoiseCov

        # Weak learner list
        self.weak_learners = []

        # Setting the current iteration m to 1
        self.cur_m = 0

        # Saving the mean of y
        self.y_mean = self.get_mean(self.Y)

        # Saving the y_mean as the most recent prediction
        self._predictions = [self.y_mean] * self.n
        self._predictions_matrix = self._predictions

        # Setting prediction covariance matrix
        self.PredictionCov = np.cov(self._predictions, self._predictions)

    @staticmethod
    def get_mean(x: list) -> float:
        """
        Calculates the mean over a list of float elements
        """
        # Initiating the sum counter
        _sum = 0

        # Infering the length of list
        _n = len(x)

        if _n == 0:
            return inf

        # Iterating through the y values
        for _x in x:
            _sum += _x

        # Returning the mean
        return _sum / _n

    def fit(
        self,
        m: int = 10
        ):
        """
        Applies the iterative algorithm
        """
        # Converting the X to suitable inputs
        _inputs = self.d[self.features].to_dict('records')

        # Iterating over the number of estimators
        for _ in range(self.cur_m+1, self.cur_m + m+1):
            # Calculating the residuals
            _residuals = [self.Y[i] - self._predictions[i] for i in range(self.n)]

            # Saving the current iterations residuals to the original dataframe
            _r_name = f"residuals"
            self.d[_r_name] = _residuals

            # Creating a weak learner
            _weak_learner = Tree(
                d = self.d.copy(),
                y_var = _r_name,
                x_vars = self.features,
                max_depth = self.max_depth,
                min_sample_leaf = self.min_sample_leaf
            )

            # Growing the tree on the residuals
            _weak_learner.fit()

            # Appending the weak learner to the list
            self.weak_learners.append(_weak_learner)

            # Getting the weak learner predictions
            _predictions_wl = [_weak_learner.predict(_x) for _x in _inputs]

            # Update prediction covariance matrix
            self.PredictionCov = np.concatenate((self.PredictionCov, np.cov(self._predictions_matrix, _predictions_wl)), axis=0)
            self.PredictionCov = np.concatenate((self.PredictionCov, np.cov(self._predictions_matrix), _predictions_wl), axis=1)

            # Setting the weak learner weight
            self.mean_predictions = np.concatenate(self.mean_predictions, self.get_mean(_predictions_wl))
            self.gamma = self.y_mean * np.linalg.inv(self.NoiseCov+self.PredictionCov).dot(self.mean_predictions)

            # Saving the current predictions
            self._predictions_matrix = np.concatenate((self._predictions_matrix, self._predictions), axis=0)
            self._predictions = [self._predictions[i] + self.gamma[_] * _predictions_wl[i] for i in range(self.n)]

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, x: dict) -> float:
        """
        Given the dictionary, predict the value of the y variable
        """
        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), self.NoiseCov, len(x))

        # Starting from the mean
        yhat = self.y_mean + pred_noise[0, :]

        # And aggregating predictions
        for _m in range(self.cur_m):
            yhat += self.gamma[_m+1] * (self.weak_learners[_m].predict(x) + pred_noise[0, _m+1])
        return yhat
