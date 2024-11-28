# The base class with the weak learner 
# from GradientBoosting.tree import Tree
from sklearn.tree import DecisionTreeRegressor as Tree
import scipy as sp
import RobustIntegration.auxilliaryFunctions as auxfun

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
        TrainNoiseCov: float = 0,
    ):

        # Saving the train dataset
        self.X, self.y = X, y

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate and coefficients (weights)
        self.gamma = []  # np.array([1])

        # Saving the noise covariance matrix
        self.TrainNoiseCov = TrainNoiseCov

        # Weak learner list
        self.weak_learners = []

        # Setting the current iteration m to 1
        self.cur_m = 0

        # Optimizing the first regressor gamma_0 and saving as the most recent prediction
        def reg0_line_search(self):
            miny, maxy = np.min(self.y), np.max(self.y)
            npts = int(1e3)
            g_vec = np.linspace(miny, maxy, num=int(npts))
            G = np.repeat(g_vec.reshape(1, npts), len(y), axis=0)
            Y = np.repeat(y, npts, axis=1)
            if self.TrainNoiseCov[0, 0] == 0:
                # deriv = -np.sign(G-Y).sum()
                cost_mat = np.abs(G-Y)
            else:
                s0 = np.sqrt(self.TrainNoiseCov[0, 0])
                G_Y_s0 = (G-Y) / s0
                cost_mat = np.sqrt(2/np.pi) * s0 * np.exp(-0.5 * G_Y_s0**2) + (G-Y) * (1 - 2*sp.stats.norm.cdf(-G_Y_s0))
                # deriv = -np.sqrt(2/np.pi) * G_Y_s0*s0 * gaussExpTerm * (1-2*sp.stats.norm.cdf(-G_Y_s0)) + 2*G_Y_s0 * gaussExpTerm/np.sqrt(2*np.pi*s0**2)
            # g_idx = np.argmin(np.abs(deriv.sum(axis=0)))
            cost = cost_mat.sum(axis=0)
            g_idx = np.argmin(cost)

            # - - - - - - - - - - - - - - - - - - - - - - - - - -
            if False:
                fig_lae = plt.figure(figsize=(12, 8))
                plt.plot(g_vec, 10 * np.log10(cost), '.', label="LAE")
                plt.xlabel('gamma')
                plt.ylabel('LAE [dB]')
                plt.legend()
                plt.show(block=False)
                plt.close(fig_lae)
            # - - - - - - - - - - - - - - - - - - - - - - - - - -

            return g_vec[g_idx]

        n_samples = len(y)
        self.reg0 = reg0_line_search(self)
        self._predictions_wl = self.reg0 * np.ones(n_samples).reshape(n_samples, 1)  # initialize 1st regressor predictions

        # Saving predictions of latest weak-learner (for coefficient calculation)
        self._predictions = self._predictions_wl

        # Initialize residuals
        self._residuals = self._predictions - y

    ''' - - - LAE GradBoost: Unconstrained weight optimization - - - '''
    def fit(self, X, y, m: int = 10):
        """
        Train ensemble members using Robust GradientBoosting
        """

        # Iterating over the number of estimators
        for _ in range(self.cur_m+1, self.cur_m+m+1):
            # Growing the tree on the residuals
            _weak_learner = Tree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_sample_leaf,
                criterion="absolute_error"
            )
            _weak_learner.fit(X, -np.sign(self._residuals))
            self.weak_learners.append(_weak_learner)  # Appending the weak learner to the list

            # Getting the weak learner predictions
            self._predictions_wl = _weak_learner.predict(X).reshape(len(y), 1)

            # Setting the weak learner weight
            gamma_init, sigma = np.array([[1.0]]), np.sqrt(self.TrainNoiseCov[_, _])

            def calc_args(_predictions_wl, _residuals, gamma, sigma):
                """ This function calculates the arguments a,b,c,d in the cost function w.r.t. gamma_t"""
                mu = gamma * _predictions_wl + _residuals
                if sigma == 0.0:  # FoldedNormal analysis requires sigma>0
                    a, b, d = 0.0, 0.0, 1.0
                    c = np.abs(mu)
                else:
                    gs = np.abs(gamma) * np.abs(sigma)
                    a = np.sqrt(2 / np.pi) * gs
                    b = np.exp(-0.5 * mu ** 2 / gs ** 2)
                    d = 1 - 2 * sp.stats.norm.cdf(-mu / gs)
                    c = mu
                return a, b, c, d

            def grad_gb_mae(_predictions_wl, _residuals, gamma, sigma):
                """ This function calculates the gradient of the cost function w.r.t. gamma_t"""
                a, b, c, d = calc_args(_predictions_wl, _residuals, gamma, sigma)
                phi = _predictions_wl
                mu = gamma * phi + _residuals
                if sigma == 0.0:  # FoldedNormal analysis requires sigma>0
                    grad = np.mean(phi * np.sign(mu), axis=0, keepdims=True)
                else:
                    gs = np.abs(gamma) * np.abs(sigma)
                    a_tag = np.sqrt(2 / np.pi) * sigma * np.sign(gamma)
                    c_tag = phi
                    b_tag = -mu * (phi * gamma - mu) / (gamma ** 3 * sigma ** 2) * np.exp(-0.5 * (mu / gs) ** 2)
                    d_tag = 2 * sp.stats.norm.pdf(-0.5 * mu / gs ** 2) * (phi * gs - np.sign(gamma) * sigma * mu) / (
                                gs ** 2)
                    grad = np.mean(a_tag * b + b_tag * a + c_tag * d + d_tag * c, axis=0, keepdims=True)
                return grad

            def cost_gb_mae(_predictions_wl, _residuals, gamma, sigma):
                a, b, c, d = calc_args(_predictions_wl, _residuals, gamma, sigma)
                return np.mean(a * b + c * d)

            grad_fun = lambda gamma: grad_gb_mae(self._predictions_wl, self._residuals, gamma, sigma)
            cost_fun = lambda gamma: cost_gb_mae(self._predictions_wl, self._residuals, gamma, sigma)
            cost_evolution, gamma_evolution, stop_iter = auxfun.gradient_descent_scalar(gamma_init, grad_fun, cost_fun,
                                                                               max_iter=1000, min_iter=100,
                                                                               tol=1e-12, learn_rate=0.3, decay_rate=0.2)
            # cost_evolution, gamma_evolution, stop_iter = self.gradient_descent(gamma_init, sigma, max_iter=1000, min_iter=100, tol=1e-12, learn_rate=0.3, decay_rate=0.2)
            new_gamma = gamma_evolution[np.argmin(cost_evolution[0:stop_iter])]

            # - - - - - - - - - - - - -
            if False:
                fig_debug1 = plt.figure()
                plt.plot(range(0, stop_iter, 1), 10*np.log10(cost_evolution[0:stop_iter]), label='Cost', linestyle='-', marker='o', color='blue')
                plt.plot(np.argmin(cost_evolution[0:stop_iter]), 10*np.log10(np.min(cost_evolution[0:stop_iter])), label='Optimum', marker='x', color='red')
                plt.legend(loc="upper right", fontsize=12)
                plt.xlabel("GD iteration", fontsize=14)
                plt.ylabel("LAE (Cost function)", fontsize=14)
                plt.title("sigma: " + "{:.2f}".format(sigma) + ", _m=" + "{:d}".format(_) + ", gamma=" + "{:.4f}".format(new_gamma[0, 0]))
                plt.grid()
                plt.show(block=False)
                plt.pause(0.05)

                fig_debug2 = plt.figure()
                plt.plot(range(0, stop_iter, 1), np.concatenate(gamma_evolution[0:stop_iter], axis=0), label='gamma', linestyle='-', marker='o', color='blue')
                plt.legend(loc="upper right", fontsize=12)
                plt.xlabel("GD iteration", fontsize=14)
                plt.ylabel("gamma", fontsize=14)
                plt.grid()
                plt.show(block=False)
                plt.pause(0.05)
                plt.close(fig_debug1)
                plt.close(fig_debug2)

            if False:
                fig_dataset = plt.figure(figsize=(12, 8))
                plt.plot(self.X[:, 0], self.y, 'x', label="Train")
                plt.plot(self.X[:, 0], self._predictions, 'o', label="Prediction")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(self.get_lae(self.X, self.y))
                plt.title(np.mean(np.abs(self._predictions-self.y)))
                plt.legend()
                plt.show(block=False)
                plt.close(fig_dataset)
            # - - - - - - - - - - - - -

            # Adding new weight to list
            if _ == 1:
                self.gamma = np.array(new_gamma)
            else:
                self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)

            # Saving the current predictions
            self._predictions = self._predictions + new_gamma * self._predictions_wl

            # Updating the residuals
            self._residuals = np.subtract(self._predictions, y)

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, X, PredNoiseCov):
        """
        Given an ensemble, predict the value of the y variable for input(s) X
        """
        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), PredNoiseCov[:self.cur_m+1, :self.cur_m+1], X.shape[0])

        # Starting from the (noisy) mean
        yhat = self.reg0 + pred_noise[:, 0]

        # And aggregating (noisy) predictions
        for _m in range(self.cur_m):
            noisy_pred = self.weak_learners[_m].predict(X) + pred_noise[:, _m+1]
            yhat += self.gamma[_m] * noisy_pred
        return yhat

    def get_lae(self, X, y):
        """
        Calculate the LAE of the predictions made by an ensemble for arbitrary input(s) X
        """
        return np.abs(np.subtract(y, self.predict(X))).mean()