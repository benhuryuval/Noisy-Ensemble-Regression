import numpy as np
import scipy as sp

class BaggingRobustRegressor:
    def __init__(self, bagging_regressor, noise_covariance, n_base_estimators, integration_type='bem'):
        self.bagging_regressor = bagging_regressor
        if noise_covariance is None:
            self.noise_covariance = np.eye(n_base_estimators)
        else:
            self.noise_covariance = noise_covariance
        self.weights = np.ones([n_base_estimators,]) / n_base_estimators
        self.n_base_estimators = n_base_estimators
        self.integration_type = integration_type

    def fit(self, X, y):
        # train bagging regressor
        self.bagging_regressor.fit(X, y)
        # calculate integration weights
        if self.integration_type == 'bem':
            self.weights = np.ones([self.n_base_estimators,]) / self.n_base_estimators
        elif self.integration_type == 'gem':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            error_covariance = np.cov(base_prediction - y)  # cov[\hat{f}-y]
            if False: #np.all(np.linalg.eigvals(error_covariance) > 0):  # positive definite
                v_min = np.linalg.inv(error_covariance).dot(np.ones([self.n_base_estimators, 1]))
                self.weights = v_min.T / v_min.sum()
            elif np.all(np.linalg.eigvals(error_covariance) >= 0):  # positive semi-definite
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(error_covariance, ones_mat)  # eigenvectors of cov[\hat{f}-y]
                min_w = np.min(np.abs(w.real))
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                self.weights = v_min.T / v_min.sum()
                # self.weights = v[:, min_w_idxs[0]] / v[:, min_w_idxs[0]].sum()
            else:
                print("Error: Invalid covariance matrix.")
                self.weights = []
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
            if False: #np.all(np.linalg.eigvals(c_mat) > 0):  # positive definite
                v_min = np.linalg.inv(c_mat).dot(np.ones([self.n_base_estimators, 1]))
                self.weights = v_min.T / v_min.sum()
            elif np.all(np.linalg.eigvals(error_covariance) >= 0):  # positive semi-definite
                ones_mat = np.ones([self.n_base_estimators, self.n_base_estimators])
                w, v = sp.linalg.eig(c_mat, ones_mat)
                min_w = np.min(w.real)
                min_w_idxs = [index for index, element in enumerate(w) if min_w == element]
                v_min = v[:, min_w_idxs].mean(axis=1)
                if "num_M="+str(min_w_idxs.__len__())>1:
                    a=1
                self.weights = v_min.T / v_min.sum()
            else:
                ValueError('Invalid covariance matrix')
        elif self.integration_type == 'robust-lr':
            base_prediction = np.zeros([self.n_base_estimators, len(y)])
            for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
                base_prediction[k, :] = base_estimator.predict(X)
            self.weights = np.linalg.inv(base_prediction.dot(base_prediction.T)/len(y) + self.noise_covariance).dot(base_prediction).dot(y)/len(y)  # least-squares
        else:
            ValueError('Invalid integration type.')
        return self

    def predict(self, X, weights=None):
        if weights is None:
            weights = self.weights
        # obtain base predictions
        base_prediction = np.zeros([self.n_base_estimators, len(X)])
        for k, base_estimator in enumerate(self.bagging_regressor.estimators_):
            base_prediction[k, :] = base_estimator.predict(X)
        # add noise
        pred_noise = np.random.multivariate_normal(np.zeros(self.n_base_estimators), self.noise_covariance, len(X))
        base_prediction = base_prediction + pred_noise.T
        # return integrated noisy predictions
        return weights.dot(base_prediction).T
