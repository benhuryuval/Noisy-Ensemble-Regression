import numpy as np
import scipy as sp
import robustRegressors.auxilliaryFunctions as auxfun

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


    def __init__(self, bagging_regressor, noise_covariance=None, n_base_estimators=1, integration_type='bem'):
        self.bagging_regressor = bagging_regressor
        if noise_covariance is None:
            self.noise_covariance = np.eye(n_base_estimators)
        else:
            self.noise_covariance = noise_covariance
        self.weights = np.ones([n_base_estimators,]) / n_base_estimators
        self.n_base_estimators = n_base_estimators
        self.integration_type = integration_type

    def fit_mse(self, X, y):

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

            if auxfun.is_psd_mat(error_covariance):
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(error_covariance, ones_mat)  # eigenvectors of cov[\hat{f}-y]
                min_w = np.min(np.abs(w.real))
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
            else:
                print("Error: Invalid covariance matrix.")
                ValueError('Invalid covariance matrix')
                self.weights = None

        elif self.integration_type == 'lr':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            self.weights = np.linalg.inv(base_prediction.dot(base_prediction.T)).dot(base_prediction).dot(y)  # least-squares

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
            if auxfun.is_psd_mat(error_covariance):
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(c_mat, ones_mat)
                min_w = np.min(w.real)
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
            else:
                print('Invalid covariance matrix.')
                ValueError('Invalid covariance matrix')
                self.weights = None

        elif self.integration_type == 'robust-lr':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            self.weights = np.linalg.inv(base_prediction.dot(base_prediction.T)/len(y) + self.noise_covariance).dot(base_prediction).dot(y)/len(y)  # least-squares

        else:
            print('Invalid integration type.')
            ValueError('Invalid integration type.')
            self.weights = None

        return self

    def fit_mae(self, X, y):

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

            if auxfun.is_psd_mat(error_covariance):
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(error_covariance, ones_mat)  # eigenvectors of cov[\hat{f}-y]
                min_w = np.min(np.abs(w.real))
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
            else:
                print("Error: Invalid covariance matrix.")
                self.weights = None

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
                sigma = np.sqrt(alpha.dot(noise_cov.dot(alpha.T)))

                mu_tag = base_prediction
                sig_tag = (noise_cov.dot(alpha.T)) / sigma
                mu_ovr_sig_tag = (base_prediction*sigma - mu*sig_tag) / (sigma**2)
                b_tag = -1 * np.exp(-0.5 * (mu/sigma)**2) * (mu/sigma) * mu_ovr_sig_tag
                d_tag = 2*sp.stats.norm.pdf(-mu/sigma) * mu_ovr_sig_tag

                grad = np.sqrt(2/np.pi)*sig_tag*np.exp(-0.5 * (mu/sigma)**2) +  \
                    np.sqrt(2 / np.pi) * sigma * b_tag +                        \
                    mu_tag * (1 - 2*sp.stats.norm.cdf(-mu/sigma)) +             \
                    mu*d_tag
                return grad.mean(1)

            def cost_rgem_mae(alpha, noise_cov, base_prediction, y):
                mu = alpha.dot(base_prediction) - y
                sigma = np.sqrt(alpha.dot(noise_cov.dot(alpha.T)))

                cost = np.sqrt(2/np.pi)*sigma*np.exp(-0.5 * (mu/sigma)**2) +  \
                    mu * (1 - 2*sp.stats.norm.cdf(-mu/sigma))
                return cost.mean(1)

            grad_fun = lambda weights: grad_rgem_mae(weights, self.noise_covariance, base_prediction, y)
            cost_fun = lambda weights: cost_rgem_mae(weights, self.noise_covariance, base_prediction, y)
            cost_evolution, weights_evolution, stop_iter = auxfun.gradient_descent(weights_init, grad_fun, cost_fun,
                                                                               max_iter=5000, min_iter=100,
                                                                               tol=1e-6, learn_rate=0.01, decay_rate=0.2)
            self.weights = weights_evolution[np.argmin(cost_evolution[0:stop_iter])]

        else:
            print('Invalid integration type.')
            ValueError('Invalid integration type.')
            self.weights = None

        return self

    def fit(self, X, y):
        if self.error == "mse":
            self.fit_mse(X, y)
        elif self.error == "mae":
            self.fit_mae(X, y)
        return self

    def predict(self, X, weights=None):
        if weights is None:
            weights = self.weights

        # Obtain base predictions from ensemble
        base_prediction = np.zeros([self.n_base_estimators, len(X)])
        for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
            base_prediction[k, :] = base_estimator.predict(X)

        # Generate noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.n_base_estimators), self.noise_covariance, len(X))

        # Return integrated noisy predictions
        return weights.dot(base_prediction + pred_noise.T).T