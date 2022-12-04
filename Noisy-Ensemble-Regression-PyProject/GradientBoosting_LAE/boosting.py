# The base class with the weak learner 
# from GradientBoosting.tree import Tree
from sklearn.tree import DecisionTreeRegressor as Tree
import scipy as sp

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

        # Optimizing the first regressor gamma_0 and saving as the most recent prediction
        def reg0_line_search(self):
            miny, maxy = np.min(self.y), np.max(self.y)
            npts = int(1e3)
            g0 = np.linspace(miny, maxy, num=int(npts))
            G = np.repeat(g0.reshape(1, npts), len(y), axis=0)
            Y = np.repeat(y, npts, axis=1)
            s0 = self.NoiseCov[0, 0]
            G_Y_s0 = (G-Y) / s0
            gaussExpTerm = np.exp(-0.5 * G_Y_s0**2)
            deriv = -np.sqrt(2/np.pi) * G_Y_s0*s0 * gaussExpTerm * (1-2*sp.stats.norm.cdf(-G_Y_s0)) + 2*G_Y_s0 * gaussExpTerm/np.sqrt(2*np.pi*s0**2)
            g0idx = np.argmin(np.abs(deriv.sum(axis=0)))
            return g0[g0idx]

        n_samples = len(y)
        self.reg0 = reg0_line_search(self)
        self._predictions = self.reg0 * np.ones(n_samples).reshape(n_samples, 1)  # initialize 1st regressor predictions

        # Saving predictions of latest weak-learner (for coefficient calculation)
        self._predictions_wl = self._predictions

        # Initialize residuals
        self._residuals = y - self._predictions

    ''' - - - LAE GradBoost: Unconstrained weight optimization - - - '''
    def grad_gamma(self, gamma, sigma):
        """ This function calculates the gradient of the cost function w.r.t. gamma_t"""
        phi = self._predictions
        mu = gamma*phi - self._residuals

        gs = np.abs(gamma) * np.abs(sigma)
        gs2 = gs**2

        a = np.sqrt(2*gs2/np.pi)
        b = np.exp(-0.5*mu**2/gs2)
        c = mu
        d = 1 - 2*sp.stats.norm.cdf(-mu/np.abs(gs))

        a_tag = np.sqrt(2/np.pi)*sigma*np.sign(gamma)
        b_tag = -mu * (phi*gamma - mu)/(gamma**3 * sigma**2) * np.exp(-0.5*(mu**2)/(gamma**2*sigma**2))
        c_tag = phi
        d_tag = 2 * sp.stats.norm.pdf(-0.5*mu/gs2) * (phi*gs - np.sign(gamma)*sigma*mu)/gs2

        return np.mean(a_tag*b + b_tag*a + c_tag*d + d_tag*c, axis=0, keepdims=True)

    def gradient_descent(self, gamma_init, sigma, max_iter=30000, min_iter=10, tol=1e-5, learn_rate=0.2, decay_rate=0.2):
        """ This function calculates optimal coefficients with gradient descent method using an early stop criteria and
        selecting the minimal value reached throughout the iterations """
        # initializations
        cost_evolution = [None]*max_iter
        gamma_evolution = [None]*max_iter
        eps = 1e-8  # tolerance value for adagrad learning rate update
        # first iteration
        gamma_evolution[0] = gamma_init
        cost_evolution[0] = self.get_training_error(self.X, self.y, gamma_init)
        step, i = 0, 0  # initialize gradient-descent step to 0, iteration index in evolution
        # perform gradient-descent
        for i in range(1, max_iter):
            # calculate grad, update momentum and alpha
            grad = self.grad_gamma(gamma_evolution[i - 1], sigma)
            # update learning rate and advance according to AdaGrad
            learn_rate_upd = np.divide(gamma_evolution[i - 1] * learn_rate, grad + eps)
            step = decay_rate * step - learn_rate_upd.dot(grad)
            gamma_evolution[i] = gamma_evolution[i-1] + step
            # update cost status and history for early stop
            cost_evolution[i] = self.get_training_error(self.X, self.y, gamma_evolution[i])
            # check convergence
            if i > min_iter and np.abs(cost_evolution[i]-cost_evolution[i-1]) <= tol:
                break
        return cost_evolution, gamma_evolution, i

    def fit(self, X, y, m: int = 10):
        """
        Train ensemble members using Robust GradientBoosting
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
            self._predictions_wl = _weak_learner.predict(X).reshape(len(y), 1)

            # Setting the weak learner weight
            gamma_init, sigma = 1, self.NoiseCov[_, _]
            cost_evolution, gamma_evolution, stop_iter = self.gradient_descent(gamma_init, sigma, max_iter=250, min_iter=10, tol=1e-5, learn_rate=0.1, decay_rate=0.2)
            new_gamma = gamma_evolution[np.argmin(cost_evolution[0:stop_iter])]

            # Adding new weight to list
            if _ == 1:
                self.gamma = new_gamma
            else:
                self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)

            # Saving the current predictions
            self._predictions = self._predictions + new_gamma * self._predictions_wl

            # Updating the residuals
            self._residuals = np.subtract(y, self._predictions)

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, X):
        """
        Given an ensemble, predict the value of the y variable for input(s) X
        """
        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), self.NoiseCov, X.shape[0])

        # Starting from the (noisy) mean
        yhat = self.reg0 + pred_noise[:, 0]

        # And aggregating (noisy) predictions
        for _m in range(self.cur_m):
            noisy_pred = self.weak_learners[_m].predict(X) + pred_noise[:, _m+1]
            yhat += self.gamma[_m] * noisy_pred
        return yhat

    def get_training_error(self, X, y, new_gamma):
        """
        Calculate the MSE of the predictions made by an ensemble for input(s) X
        """
        y_hat = self._predictions + new_gamma * self._predictions_wl
        return np.sqrt(np.square(np.subtract(y[:, 0], y_hat)).mean())


