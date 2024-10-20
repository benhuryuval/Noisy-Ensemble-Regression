import numpy as np
import scipy as sp
import robustRegressors.auxilliaryFunctions as auxfun
from multivariate_laplace import multivariate_laplace

class rBaggReg:  # Robust Bagging Regressor
    # This class contains the implementation of Bagging aggregation methods for a given regression
    # ensemble and prediction noise covariance matrix.
    #
    # The options are:
    #   BEM / rBEM - Basic ensemble method.
    #   GEM / rBEM - Generalized ensemble method.
    #   LR / rLR - Linear regressor (least-squares solution of coefficients allocation problem)
    #
    # The supported error criteria are: MSE (l2) and MAE (l1).


    def __init__(self, bagging_regressor, noise_covariance=None, n_base_estimators=1, integration_type='bem', gd_tol=1e-6, learn_rate= 1e-6, decay_rate=0.0, bag_tol=0):
        self.bagging_regressor = bagging_regressor
        if noise_covariance is None:
            self.noise_covariance = np.eye(n_base_estimators)
        else:
            self.noise_covariance = noise_covariance
        self.weights = np.ones([n_base_estimators,]) / n_base_estimators
        self.n_base_estimators = n_base_estimators
        self.integration_type = integration_type
        # gradient-descent params
        self.gd_tol, self.learn_rate, self.decay_rate = gd_tol, learn_rate, decay_rate
        self.bag_tol = bag_tol

    def fit_mse(self, X, y, lamda=1):

        # Train bagging regressor
        self.bagging_regressor.fit(X, y)

        # Calculate aggregation coefficients
        if self.integration_type == 'bem':
            self.weights = np.ones([self.n_base_estimators,]) / self.n_base_estimators

        elif self.integration_type == 'gem':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            error_covariance = np.cov(base_prediction - y)  # cov[\hat{f}-y]
            # err_mat_rglrz = np.real_if_close(error_covariance, tol=1e-1) + 1e-1 * np.diag(np.ones(self.n_base_estimators, ))
            err_mat_rglrz = error_covariance + self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))

            if auxfun.is_psd_mat(err_mat_rglrz):
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(err_mat_rglrz, ones_mat)  # eigenvectors of cov[\hat{f}-y]
                min_w = np.min(np.abs(w.real))
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
            else:
                print("Error: Invalid covariance matrix.")
                raise ValueError('Invalid covariance matrix')
                self.weights = None

        elif self.integration_type == 'lr':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            c_mat = base_prediction.dot(base_prediction.T)/len(y)
            reg_mat = self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))
            self.weights = np.linalg.inv(c_mat + reg_mat).dot(base_prediction).dot(y)/len(y)  # least-squares

        elif self.integration_type == 'robust-bem':
            w, v = sp.linalg.eig(self.noise_covariance)
            min_w = np.min(w.real)
            min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
            self.weights = v[:, min_w_idxs].mean(axis=1) / v[:, min_w_idxs].mean(axis=1).sum()

        elif self.integration_type == 'robust-gem':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            error_covariance = np.cov(base_prediction - y)  # cov[\hat{f}-y]
            c_mat = error_covariance + self.noise_covariance
            # c_mat_rglrz = np.real_if_close(c_mat, tol=1e-9*c_mat.mean()) + self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))
            c_mat_rglrz = c_mat + self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))
            if auxfun.is_psd_mat(c_mat_rglrz):
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(c_mat_rglrz, ones_mat)
                min_w = np.min(w.real)
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
            else:
                print('Invalid covariance matrix.')
                raise ValueError('Invalid covariance matrix')
                self.weights = None

        elif self.integration_type == 'robust-lr':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            c_mat = base_prediction.dot(base_prediction.T)/len(y)
            ncov_mat = self.noise_covariance
            reg_mat = self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))
            self.weights = np.linalg.inv(c_mat + lamda*ncov_mat + reg_mat).dot(base_prediction).dot(y)/len(y)  # least-squares
            # self.scores = np.mean((base_prediction * np.expand_dims(self.weights, axis=1) - y)**2, axis=1)
            self.scores = np.mean((base_prediction - y)**2, axis=1)

        else:
            print('Invalid integration type.')
            raise ValueError('Invalid integration type.')
            self.weights = None

        return self

    def fit_mae(self, X, y):

        # Train bagging regressor
        self.bagging_regressor.fit(X, y)

        # Calculate aggregation coefficients
        if self.integration_type == 'bem':
            self.weights = np.ones([self.n_base_estimators,]) / self.n_base_estimators

        elif self.integration_type == 'gem':
            # Getting the weak learner predictions
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)

            # Setting the weak learner weight via gradient-descent optimization
            weights_init = np.array([np.ones([self.n_base_estimators, ])])/self.n_base_estimators
            def grad_gem_mae(alpha, base_prediction, y):
                grad = base_prediction * np.sign(alpha.dot(base_prediction) - y)
                return grad.mean(1)

            def cost_gem_mae(alpha, base_prediction, y):
                cost = np.abs(alpha.dot(base_prediction) - y)
                return cost.mean(1)

            grad_fun = lambda weights: grad_gem_mae(weights, base_prediction, y)
            cost_fun = lambda weights: cost_gem_mae(weights, base_prediction, y)
            cost_evolution, weights_evolution, stop_iter = auxfun.gradient_descent(weights_init, grad_fun, cost_fun,
                                                                               max_iter=5000, min_iter=500,
                                                                               tol=self.gd_tol, learn_rate=self.learn_rate, decay_rate=self.decay_rate)
            self.weights = weights_evolution[np.argmin(cost_evolution[0:stop_iter])]

            # # DEBUG # #
            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.figure("Cost evolution", figsize=(8, 6), dpi=300), plt.axes()
                plt.plot(cost_evolution[0:stop_iter])
                plt.xlabel('Iteration', fontsize=18)
                plt.ylabel("Cost", fontsize=18)
                plt.show(block=False)
                bb=0
            # # DEBUG END # #

        elif self.integration_type == 'robust-bem':
            w, v = sp.linalg.eig(self.noise_covariance)
            min_w = np.min(w.real)
            min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
            self.weights = v[:, min_w_idxs].mean(axis=1) / v[:, min_w_idxs].mean(axis=1).sum()

        elif self.integration_type == 'robust-gem':
            # Getting the weak learner predictions
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)

            # Setting the weak learner weight via gradient-descent optimization
            weights_init = np.array([np.ones([self.n_base_estimators, ])])/self.n_base_estimators
            def grad_rgem_mae(alpha, noise_cov, base_prediction, y):
                mu = alpha.dot(base_prediction) - y
                mu_tag = base_prediction
                sigma = np.sqrt(alpha.dot(noise_cov.dot(alpha.T)))

                if sigma == 0.0:
                    grad = mu_tag * np.sign(mu)
                else:
                    rho = mu / sigma
                    sig_tag = (noise_cov.dot(alpha.T)) / sigma
                    rho_tag = (base_prediction*sigma - mu*sig_tag) / (sigma**2)
                    grad = np.sqrt(2/np.pi) * np.exp(-0.5 * rho**2) * (sig_tag - sigma*rho*rho_tag) +  \
                        mu_tag * (1 - 2*sp.stats.norm.cdf(-rho)) +             \
                        2 * mu * rho_tag * sp.stats.norm.pdf(-rho)
                return grad.mean(1)

            def cost_rgem_mae(alpha, noise_cov, base_prediction, y):
                mu = alpha.dot(base_prediction) - y
                sigma = np.sqrt(alpha.dot(noise_cov.dot(alpha.T)))

                if sigma == 0.0:
                    cost = mu * (1 - 2*np.sign(-mu))
                else:
                    cost = np.sqrt(2/np.pi)*sigma*np.exp(-0.5 * (mu/sigma)**2) +  \
                        mu * (1 - 2*sp.stats.norm.cdf(-mu/sigma))
                return cost.mean(1)

            grad_fun = lambda weights: grad_rgem_mae(weights, self.noise_covariance, base_prediction, y)
            cost_fun = lambda weights: cost_rgem_mae(weights, self.noise_covariance, base_prediction, y)
            cost_evolution, weights_evolution, stop_iter = auxfun.gradient_descent(weights_init, grad_fun, cost_fun,
                                                                               max_iter=5000, min_iter=500,
                                                                               tol=self.gd_tol, learn_rate=self.learn_rate, decay_rate=self.decay_rate)
            self.weights = weights_evolution[np.argmin(cost_evolution[0:stop_iter])]
            # self.weights = self.weights / np.sum(self.weights)  # TODO: this is just to debug lower bound

            # # DEBUG # #
            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.figure(figsize=(8, 6), dpi=300), plt.axes()
                plt.plot(cost_evolution[0:stop_iter], label="Cost evolution",)
                plt.xlabel('Iteration', fontsize=18)
                plt.ylabel("Cost", fontsize=18)
                plt.show(block=False)
                bb=0
                plt.close(fig)
            # # DEBUG END # #

        else:
            print('Invalid integration type.')
            raise ValueError('Invalid integration type.')
            self.weights = None

        return self

    def fit(self, X, y):
        if self.error == "mse":
            self.fit_mse(X, y)
        elif self.error == "mae":
            self.fit_mae(X, y)
        return self

    def predict(self, X, weights=None, rng=np.random.default_rng(seed=42), noiseless=False, noisetype='gaussian'):
        if weights is None:
            weights = self.weights

        # Obtain base predictions from ensemble
        base_prediction = np.zeros([self.n_base_estimators, len(X)])
        for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
            base_prediction[k, :] = base_estimator.predict(X)

        # Generate noise
        if noisetype == 'gaussian':
            pred_noise = rng.multivariate_normal(np.zeros(self.n_base_estimators), self.noise_covariance, len(X))
        elif noisetype == 'laplace':
            pred_noise = multivariate_laplace.rvs(np.zeros(self.n_base_estimators), self.noise_covariance, len(X))
        else:
            raise ValueError('Invalid noise type')
        if noiseless:
            pred_noise *= 0

        # Return integrated noisy predictions
        return weights.dot(base_prediction + pred_noise.T).T

    def calc_mae_lb(self, X, y, weights=None):
        # Train bagging regressor, if needed
        if (self.weights == None).any():
            self.bagging_regressor.fit(X, y)

        # Obtain base predictions from ensemble
        base_prediction = np.zeros([self.n_base_estimators, len(X)])
        for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
            base_prediction[k, :] = base_estimator.predict(X)
        # Calculate lower bound
        mae_cln = auxfun.calc_error(weights.dot(base_prediction).T, y, 'mae')  # clean mae with non-robust weights

        mu = np.abs(base_prediction.transpose() - y)
        mu_max = np.max(mu, axis=1)
        mu_min = np.min(mu, axis=1)

        c_mat = self.noise_covariance
        if auxfun.is_psd_mat(c_mat):
            ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
            w, v = sp.linalg.eig(c_mat, ones_mat)
            sigma_bar = np.sqrt(np.nanmin(w.real))
            sigma_max = np.sqrt(np.nanmax(w.real))
            diff = np.sqrt(2 / np.pi) * sigma_bar - mu_max
            diff_ind = (np.sign(diff) + 1) / 2
        else:
            self.mae_lb = None
            print('Invalid covariance matrix.')
            raise ValueError('Invalid covariance matrix')

        self.mae_lb = mae_cln, mae_cln + np.nanmean(diff * np.exp(-1/2 * diff_ind * (mu_max/sigma_bar)**2 -1/2 * (1-diff_ind) * (mu_min/sigma_max)**2))
        return self.mae_lb

    def calc_mae_ub(self, X, y):
        # Obtain base predictions from ensemble
        base_prediction = np.zeros([self.n_base_estimators, len(X)])
        for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
            base_prediction[k, :] = base_estimator.predict(X)

        # # Upper bound using BEM # #
        weights = np.ones([self.bagging_regressor.n_estimators, ]) / self.bagging_regressor.n_estimators
        mae_cln = auxfun.calc_error(weights.dot(base_prediction).T, y, 'mae')  # clean mae with non-robust weights
        mae_ub_bem = mae_cln + np.sqrt(2 / np.pi * self.noise_covariance.sum())

        # # Upper bound using GEM # #
        # Obtain coefficients
        err_mat_rglrz = self.noise_covariance  # + self.bag_tol * np.diag(np.ones(self.n_base_estimators, ))
        if auxfun.is_psd_mat(err_mat_rglrz):
            w, v = sp.linalg.eig(err_mat_rglrz)  # eigenvectors of cov[\hat{f}-y]
            min_w = np.nanmin(w.real)
            min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
            v_min = v[:, min_w_idxs].mean(axis=1)
            if (self.weights == None).any():
                weights = v_min.T / v_min.sum()
        else:
            print("Error: Invalid covariance matrix.")
            raise ValueError('Invalid covariance matrix')
        mae_cln = auxfun.calc_error(weights.dot(base_prediction).T, y, 'mae')  # clean mae with non-robust weights
        mae_ub_gem = mae_cln + np.sqrt(2 / np.pi * min_w)

        # Set upper bound(s)
        self.mae_ub = np.nanmin([mae_ub_bem, mae_ub_gem])
        return mae_ub_bem, mae_ub_gem