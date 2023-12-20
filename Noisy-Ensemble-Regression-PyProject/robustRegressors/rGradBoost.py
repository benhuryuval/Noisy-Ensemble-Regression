# The base class with the weak learner 
from sklearn.tree import DecisionTreeRegressor as Tree
import robustRegressors.auxilliaryFunctions as auxfun

# Data wrangling 
import pandas as pd
import scipy as sp
import numpy as np

# Plotting
import matplotlib.pyplot as plt

class rGradBoost:
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
        RobustFlag: bool = False,
        criterion: str = "mse",
        gd_tol=1e-6, gd_learn_rate=1e-6, gd_decay_rate=0.0
    ):

        # Saving the train dataset
        self.X, self.y = X, y

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate and coefficients (weights)
        self.gd_tol, self.gd_learn_rate, self.gd_decay_rate = gd_tol, gd_learn_rate, gd_decay_rate

        # Saving the noise covariance matrix
        self.TrainNoiseCov = TrainNoiseCov

        # Setting the robustness enable flag
        self.RobustFlag = RobustFlag

        # set the criterion(s)
        self.criterion = criterion
        if self.criterion == "mse":
            self.weak_lrnr_criterion = "mse"
        elif self.criterion == "mae":
            self.weak_lrnr_criterion = "mse"  # "absolute_error"

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
                cost_mat = np.abs(G-Y)
                # deriv = -np.sign(G-Y).sum()
            else:
                s0 = np.sqrt(self.TrainNoiseCov[0, 0])
                mu = G - Y
                cost_mat = np.sqrt(2/np.pi) * np.abs(G*s0) * np.exp(-0.5 * mu**2/s0**2) + mu * (1 - 2*sp.stats.norm.cdf(-mu/s0))
            cost = cost_mat.mean(axis=0)
            g_idx = np.argmin(cost)

            # - - - - - - - - - - - - - - - - - - - - - - - - - -
            if False:
                fig_lae = plt.figure(figsize=(12, 8))
                plt.plot(g_vec, cost_mat.mean(axis=0), '.', label="LAE")
                plt.xlabel('gamma')
                plt.ylabel('LAE [dB]')
                plt.legend()
                plt.show(block=False)
                plt.close(fig_lae)
            # - - - - - - - - - - - - - - - - - - - - - - - - - -

            return g_vec[g_idx]

        n_samples = len(y)
        if criterion == "mse":
            self.reg0 = np.mean(y) / (1 + self.TrainNoiseCov[0, 0])  # mean target value over dataset
        elif criterion == "mae":
            if self.RobustFlag:
                self.reg0 = reg0_line_search(self)
            else:
                self.reg0 = np.median(y)  # median target value over dataset

        self.gamma = [[1.0]]  # np.array([1])
        self._predictions_wl = self.reg0 * np.ones(n_samples).reshape(n_samples, 1)  # initialize 1st regressor predictions
        self._predictions_all_wl = self._predictions_wl  # initialize all regressor prediction matrix

        # Total prediction is (currently) predictions of latest (first) weak-learner
        self._predictions = self.gamma * self._predictions_wl

        # Initialize residuals
        self.criterion = criterion
        if self.criterion == "mse":
            self._residuals = y - self._predictions
        elif self.criterion == "mae":
            self._residuals = np.sign(y - self._predictions)

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
                criterion=self.weak_lrnr_criterion
            )
            _weak_learner.fit(X, self._residuals)
            self.weak_learners.append(_weak_learner)  # Appending the weak learner to the list

            # Getting the weak learner predictions
            self._predictions_wl = _weak_learner.predict(X).reshape(len(y), 1)
            self._predictions_all_wl = np.concatenate( (self._predictions_all_wl,self._predictions_wl), axis=1)

            # # DEBUG # #
            # Fit of current weak-learner
            if False:
                import matplotlib.pyplot as plt
                fig_dataset = plt.figure(figsize=(12, 8))
                plt.plot(self.X[:, 0], self._residuals, 'o', label="Train")
                plt.plot(self.X[:, 0], self._predictions_wl, '.', label="Prediction")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.close(fig_dataset)
            # # DEBUG END # #

            # Setting the weak learner weight
            if self.criterion == "mse":
                y_minus_f = np.subtract(y, self._predictions)
                sum_phi_y_minus_f = np.mean(np.multiply(self._predictions_wl, y_minus_f), keepdims=True)
                # if _ == 1:  # first weak-learner (after initialization)
                #     sum_gamma_sigma = 0.0
                # else:
                    # gamma = self.gamma.reshape(_-1, 1)
                prev_gamma = np.array(self.gamma[0:_])
                prev_CovMat = np.array(self.TrainNoiseCov[0:_, _, np.newaxis])
                sum_gamma_sigma = prev_gamma.T.dot(prev_CovMat)
                phi_sqrd = np.mean(self._predictions_wl**2, keepdims=True)
                new_gamma = (sum_phi_y_minus_f + self.RobustFlag * sum_gamma_sigma) / (phi_sqrd + self.RobustFlag * self.TrainNoiseCov[_, _])

            elif self.criterion == "mae":
                gamma_init = np.array([[1.0]])
                def grad_rgem_mae(alpha, Sigma, base_prediction, y):
                    mu = base_prediction.dot(alpha) - y
                    sigma2 = np.sqrt(alpha.T.dot(Sigma.dot(alpha)))
                    if sigma2 == 0:
                        grad = np.expand_dims(base_prediction[:, -1], axis=1) * np.sign(mu)
                    else:
                        sigma2_tag = alpha.T.dot(Sigma[:, -1])
                        mu_tag = np.expand_dims(base_prediction[:, -1], axis=1)

                        Gamma1 = np.sqrt(2/np.pi) * np.sqrt(sigma2)
                        Gamma2 = np.exp(-0.5 * mu**2/sigma2)
                        Lambda1 = mu
                        Lambda2 = 1 - 2 * sp.stats.norm.cdf(-mu/np.sqrt(sigma2))

                        Gamma1_tag = np.sqrt(2/np.pi) * sigma2_tag
                        Gamma2_tag = -1 * np.exp(-0.5 * mu**2/sigma2) * ( 2*mu*mu_tag*sigma2 - sigma2_tag*mu ) / ( 2*sigma2**2 )
                        Lambda1_tag = mu_tag
                        Lambda2_tag = 2 * sp.stats.norm.pdf(-mu/np.sqrt(sigma2)) * ( 2*mu_tag*sigma2 - sigma2_tag ) / ( 2*np.sqrt(sigma2)**3 )

                        grad = Gamma1_tag*Gamma2 + Gamma2_tag*Gamma1 + Lambda1_tag*Lambda2 + Lambda2_tag*Lambda1

                    return grad.mean(0)
                def cost_rgem_mae(alpha, noise_cov, base_prediction, y):
                    mu = base_prediction.dot(alpha) - y
                    sigma = np.sqrt(alpha.T.dot(noise_cov.dot(alpha)))
                    if sigma == 0:
                        cost = np.abs(mu)
                    else:
                        rho = mu / sigma
                        cost = np.sqrt(2 / np.pi) * sigma * np.exp(-0.5 * rho ** 2) + \
                               mu * (1 - 2 * sp.stats.norm.cdf(-rho))
                    return cost.mean(0)
                grad_fun = lambda weight: grad_rgem_mae(np.concatenate((self.gamma, weight), axis=0), self.TrainNoiseCov[0:_+1,0:_+1], self._predictions_all_wl, y)
                cost_fun = lambda weight: cost_rgem_mae(np.concatenate((self.gamma, weight), axis=0), self.TrainNoiseCov[0:_+1,0:_+1], self._predictions_all_wl, y)
                # # # # # # # # # # # # #

                gd_learn_rate = self.gd_learn_rate #* 2**(_+1)
                cost_evolution, gamma_evolution, stop_iter = auxfun.gradient_descent_scalar(gamma_init, grad_fun, cost_fun,
                                                                                            max_iter=10000, min_iter=100,
                                                                                            tol=self.gd_tol, learn_rate=gd_learn_rate,
                                                                                            decay_rate=self.gd_decay_rate)

                # # DEBUG # #
                # Cost evolution through GD
                if False:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.figure("Cost evolution", figsize=(8, 6), dpi=300), plt.axes()
                    plt.plot(cost_evolution[0:stop_iter])
                    plt.xlabel('Iteration', fontsize=18)
                    plt.ylabel("Cost", fontsize=18)
                    plt.show(block=False)
                    bb=0
                # # DEBUG END # #

                # Adding new weight to list
                new_gamma = gamma_evolution[np.nanargmin(cost_evolution[0:stop_iter])]

                # # DEBUG # #
                # Fit of aggregated prediction
                if False:
                    import matplotlib.pyplot as plt
                    fig_dataset = plt.figure(figsize=(12, 8))
                    plt.plot(self.X[:, 0], self.y, 'x', label="Train")
                    plt.plot(self.X[:, 0], self._predictions + new_gamma * self._predictions_wl, 'o', label="Prediction")
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(np.abs(np.subtract(y, self.predict(self.X, self.TrainNoiseCov[0:_ + 1, 0:_ + 1]))).mean())
                    plt.close(fig_dataset)
                # # DEBUG END # #

            # Saving the current total predictions and coefficient
            self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)
            self._predictions = self._predictions + new_gamma * self._predictions_wl

            # Updating the residuals
            # self._residuals = self._predictions - y
            if self.criterion == "mse":
                self._residuals = y - self._predictions
            elif self.criterion == "mae":
                self._residuals = np.sign(y - self._predictions)

            # - - - - - - - - - - - - -
            # LAE DEBUG PLOTS
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
                import matplotlib.pyplot as plt
                fig_dataset = plt.figure(figsize=(12, 8))
                plt.plot(self.X[:, 0], self._residuals, 'o', label="Train")
                plt.plot(self.X[:, 0], self._predictions_wl, '.', label="Prediction")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.close(fig_dataset)

                fig_dataset = plt.figure(figsize=(12, 8))
                plt.plot(self.X[:, 0], self.y, 'x', label="Train")
                plt.plot(self.X[:, 0], self._predictions, 'o', label="Prediction")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(np.abs(np.subtract(y, self.predict(self.X,self.TrainNoiseCov[0:_+1,0:_+1]))).mean())
                plt.close(fig_dataset)

                # def get_lae(self, X, y):
                #     """
                #     Calculate the LAE of the predictions made by an ensemble for arbitrary input(s) X
                #     """
                #     return np.abs(np.subtract(y, self.predict(X))).mean()
                #
                # # plt.title(self.get_lae(self.X, self.y))
                # plt.title(np.mean(np.abs(self._predictions-self.y)))
                # plt.legend()
                # plt.show(block=False)
                # plt.close(fig_dataset)
            # - - - - - - - - - - - - -

        # Incrementing the current iteration
        self.cur_m += m

    def predict(self, X, PredNoiseCov, weights=None):
        """
        Given an ensemble, predict the value of the y variable for input(s) X
        """
        if weights is None:
            weights = self.gamma

        # Generating noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.cur_m+1), PredNoiseCov[:self.cur_m+1, :self.cur_m+1], X.shape[0])

        # Starting from the (noisy) mean/median
        yhat = self.reg0 + pred_noise[:, 0]

        # And aggregating (noisy) predictions
        for _m in range(self.cur_m):
            noisy_pred = self.weak_learners[_m].predict(X) + pred_noise[:, _m+1]
            yhat += weights[_m+1] * noisy_pred
        return yhat

    def plot(self):
        import matplotlib.pyplot as plt
        fig_dataset = plt.figure(figsize=(12, 8))
        plt.plot(self.X[:, 0], self.y, 'x', label="Train")
        plt.plot(self.X[:, 0], self._predictions, 'o', label="Prediction")
        plt.xlabel('x')
        plt.ylabel('y')

    def fit_mse_noisy(self, X, y, m: int = 10):
        """
        Train ensemble members using GradientBoosting with MSE and noisy regressors (not robustly)
        """

        # Iterating over the number of estimators
        for _ in range(self.cur_m+1, self.cur_m+m+1):
            # Growing the tree on the residuals
            _weak_learner = Tree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_sample_leaf,
                criterion=self.weak_lrnr_criterion
            )
            _weak_learner.fit(X, self._residuals)
            self.weak_learners.append(_weak_learner)  # Appending the weak learner to the list

            # Getting the weak learner predictions
            self._predictions_wl = _weak_learner.predict(X).reshape(len(y), 1)
            self._predictions_all_wl = np.concatenate((self._predictions_all_wl, self._predictions_wl), axis=1)

            # # DEBUG # #
            # Fit of current weak-learner
            if False:
                import matplotlib.pyplot as plt
                fig_dataset = plt.figure(figsize=(12, 8))
                plt.plot(self.X[:, 0], self._residuals, 'o', label="Train")
                plt.plot(self.X[:, 0], self._predictions_wl, '.', label="Prediction")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.close(fig_dataset)
            # # DEBUG END # #

            # Setting the weak learner weight
            if self.criterion == "mse":
                # calculate \sum_{\tau=1}^{t-1} \alpha_\tau * \tilde{phi}_\tau
                sum_a_phi = np.zeros((X.shape[0],1))
                for _m in range(self.cur_m):
                    noise = np.random.multivariate_normal(np.zeros(self.cur_m),
                                                          self.TrainNoiseCov[_m, _m],
                                                          X.shape[0])
                    tilde_phi = self._predictions_all_wl[:, _m, np.newaxis] + noise
                    sum_a_phi += self.gamma[_m] * tilde_phi
                # calculate \alpha_t * \tilde{phi}_t
                noise = np.random.multivariate_normal(np.zeros(1),
                                                      [[self.TrainNoiseCov[self.cur_m, self.cur_m]]],
                                                      X.shape[0])
                tilde_phi_t = self._predictions_wl + noise
                # solve weight polynomial
                y_minus_f = np.subtract(y, sum_a_phi)
                C = np.mean(y_minus_f**2, keepdims=True)
                B = -2*np.mean(np.multiply(y_minus_f, tilde_phi_t), keepdims=True)
                A = np.mean(tilde_phi_t**2, keepdims=True)

                new_gamma =(-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
            else:
                raise Exception("Invalid criterion for noisy training")

            # Saving the current total predictions and coefficient
            self.gamma = np.concatenate((self.gamma, new_gamma), axis=0)
            self._predictions = self._predictions + new_gamma * self._predictions_wl

            # Updating the residuals
            self._residuals = y - self._predictions

        # Incrementing the current iteration
        self.cur_m += m
