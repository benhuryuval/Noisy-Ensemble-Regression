import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class CoefficientsOptimizer:

    def __init__(self, train_confidence_levels, noise_vars, power_limit=None, p_norm=2):
        self.trainConfidenceLevels = train_confidence_levels
        self.noiseVars = noise_vars
        self.powerLimit = power_limit
        self.pNorm = p_norm

    ''' - - - Constrained optimization of weights and gains - - - '''
    def calc_mismatch_alpha_beta(self, a, b):
        """ This function calculates the cost function of the optimization"""
        h = self.trainConfidenceLevels
        sigma = self.noiseVars

        h_1 = h.sum(axis=0, keepdims=True)
        a_G_h = a.T.dot(b*h)
        a_Sig_a = a.T.dot(np.multiply(sigma, a))
        one_H_one = np.multiply(h_1, h_1)
        a_G_H_one = np.multiply(a_G_h, h_1)
        psi = a_G_H_one / np.sqrt(one_H_one * a_Sig_a)
        return np.mean(QFunc(psi))

    def gradient_alpha_beta_projection(self, a, b):
        """
        Function that computes the gradient of the mismatch probability \tilde{P}_{\alpha,\beta}(x).
        :param a: coefficients column vector \alpha
        :param b: coefficients column vector \beta
        :param h: matrix of confidence levels column vectors h for all data samples
        :param sigma: noise variance column vector \sigma
        :return: gradient of mismatch probability with respect to \alpha and \beta calculated in (a,b)
        """
        # initializations
        h = self.trainConfidenceLevels
        sigma = self.noiseVars

        # calculate gradient w.r.t alpha
        h_1 = h.sum(axis=0, keepdims=True)
        a_G_h = a.T.dot(b*h)
        a_Sig_a = a.T.dot(np.multiply(sigma, a))
        one_H_one = np.multiply(h_1, h_1)
        a_G_H_one = np.multiply(a_G_h, h_1)
        psi = a_G_H_one / np.sqrt(one_H_one * a_Sig_a)
        psi_tag = (a_Sig_a * h_1 * (b*h) - a_G_H_one * (sigma*a)) / (a_Sig_a * np.sqrt(a_Sig_a * one_H_one))
        grad_alpha = -1/np.sqrt(2*np.pi) * np.mean(np.exp(-0.5 * psi**2) * psi_tag, axis=1, keepdims=True)

        # h_sum = h.sum(axis=0, keepdims=True)
        # # calculate constants
        # sqrt_one_h_ht_one = abs(h_sum)
        # alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        # g = ((a * b).T @ h) * h_sum / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        # # calculate gradient w.r.t alpha
        # parenthesis_term = (b * h) * h_sum / sqrt_one_h_ht_one - g * (a * sigma) / alpha_sigma_alpha
        # grad_alpha = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)

        # calculate gradient w.r.t beta
        A_H_1 = np.multiply(a*h, h_1)
        psi_tag = A_H_1 / np.sqrt(a_Sig_a * one_H_one)
        grad_beta = -1/np.sqrt(2*np.pi) * np.mean(np.exp(-0.5 * psi**2) * psi_tag, axis=1, keepdims=True)

        # parenthesis_term = (a * h) * h_sum / sqrt_one_h_ht_one / np.sqrt(alpha_sigma_alpha)
        # grad_beta = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)

        # get s by projecting gradient for beta on feasible domain
        s = np.zeros([len(grad_beta), 1])
        if grad_beta.__abs__().sum() / b.__abs__().sum() >= 1e-20:  # in case grad_beta is not zero
            s = np.sqrt(self.powerLimit / np.sum(grad_beta**2)) * grad_beta
        return grad_alpha, grad_beta, s

    # - - - Update alpha and beta together
    def frank_wolfe_joint(self, a0, b0, tol=1e-5, learn_rate=0.2, decay_rate=0.2, K_max=15000, K_min=10):
        # Initialize
        eps = 1e-8  # tolerance value for adagrad learning rate update
        a, b, rho = [None] * K_max, [None] * K_max, [None] * K_max
        da, db, cost_function = [None] * K_max, [None] * K_max, [None] * K_max
        k, a[0], b[0] = 0, a0, b0
        step_a, step_b = 0, 0
        cost_function[0] = self.calc_mismatch_alpha_beta(a[0], b[0])
        # apply the Frank-Wolfe algorithm
        for k in range(1, K_max):
            # calculate gradients w.r.t a and b
            da[k-1], tmp_db, db[k-1] = self.gradient_alpha_beta_projection(a[k-1], b[k-1]) # calculate gradient for alpha and projected gradient for beta

            # 1. update momentum and advance alpha
            learn_rate_upd = np.divide(np.eye(len(a0)) * learn_rate,
                                       np.sqrt(np.diag(np.power(np.squeeze(da[k-1]), 2))) + eps)  # update learning rate and advance according to AdaGrad
            step_a = decay_rate * step_a - learn_rate_upd.dot(da[k-1])
            a[k] = a[k-1] + step_a  # advance alpha

            # 2. advance beta
            # learn_rate_upd = np.divide(np.eye(len(b0)) * learn_rate,
            #                            np.sqrt(np.diag(np.power(np.squeeze(db[k-1]), 2))) + eps)  # update learning rate and advance according to AdaGrad
            # step_b = decay_rate * step_b - learn_rate_upd.dot(db[k-1])
            # step_b = (1-2/(2+k)) * step_b - learn_rate_upd.dot(db[k - 1])
            # b[k] = b[k-1] + step_b  # advance beta

            rho[k] = 0.25 * 2 / (2 + k)  # determine momentum/step size for beta
            b[k] = (1 - rho[k]) * b[k-1] - rho[k] * db[k-1]  # advance beta

            # b[k] = b[k-1] + db[k-1]  # advance beta

            b[k] = scale_to_constraint(b[k], self.powerLimit, self.pNorm)  # normalize beta
            # update cost function history and check convergence
            cost_function[k] = self.calc_mismatch_alpha_beta(a[k], b[k])
            if k > K_min and abs(cost_function[k] - cost_function[k-1]) <= tol: break
        # return
        return cost_function, a, b, k

    def optimize_coefficients_power(self, method='Frank-Wolfe', tol=0.0001, max_iter=15000, min_iter=10):
        n_estimators = len(self.noiseVars)
        # initialize coefficients
        a0 = np.ones([n_estimators, 1])  # initialize uniform aggregation coefficients
        # optimize coefficients and power allocation
        if method == 'Alpha-Beta-Joint':  # optimize coefficients and power allocation
            b0 = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power subject to constraint
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe_joint(a0, b0, tol=tol, K_max=max_iter, K_min=min_iter)
        if method == 'Alpha-Beta-Alternate':  # optimize coefficients and power allocation
            b0 = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power per-channel
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe_alt(a0, b0, tol=tol, K_max=max_iter, K_min=min_iter)
        elif method == 'Alpha-UniformBeta':  # optimize coefficients with uniform constrained power allocation
            beta = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power per-channel
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        elif method == 'Alpha-UnitBeta':  # optimize coefficients with unit power allocation per-channel
            beta = np.ones([n_estimators, 1])  # initialize unit power per-channel
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        # return
        return mismatch_prob, alpha, beta, stop_iter