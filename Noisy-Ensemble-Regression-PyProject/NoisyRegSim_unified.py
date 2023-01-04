# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.ensemble
import sklearn.datasets
import pandas as pd  # data handling
from sklearn.model_selection import KFold  # k-fold cross-validation
# OS traversal
import os
# System functionalities
import sys
# Adding the whole project to module paths
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import matplotlib.pyplot as plt  # Plotting
# Noisy regression classes
from GradientBoosting.boosting import RobustRegressionGB as rGradBoost_MSE  # Regression boosting MSE
from GradientBoosting_LAE.boosting import RobustRegressionGB as rGradBoost_MAE  # Regression boosting MAE
from RobustIntegration.BaggingRobustRegressor import BaggingRobustRegressor as rBaggReg_MSE
from RobustIntegration_LAE.BaggingRobustRegressor_LAE import rBaggReg as rBaggReg_MAE
import RobustIntegration.auxilliaryFunctions as aux
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constants
rng = np.random.RandomState(42)
plot_flag = False
save_results_to_file_flag = False
results_path = "Results//"

# data_type = 'white-wine'  # kc_house_data / diabetes / white-wine / sin / exp / make_reg
data_type_vec = ["kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]
test_size = 0.5  # test data fraction
KFold_n_splits = 2  # Number of k-fold x-validation dataset splits

ensemble_size = [1, 16, 64]  # Number of weak-learners
tree_max_depth = 6  # Maximal depth of decision tree
learning_rate = 0.1  # learning rate of gradient boosting
min_sample_leaf = 10

snr_db_vec = np.linspace(-40, 25, 10)  # [-10]
n_repeat = 500  # Number of iterations for estimating expected performance
sigma_profile_type = "uniform"  # uniform / good-outlier / linear

n_samples = 500  # Size of the (synthetic) dataset  in case of synthetic dataset
train_noise = 0.1  # Standard deviation of the measurement / training noise in case of synthetic dataset

criterion = "mse"  # "mse" / "mae"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Main simulation loop(s)
####################################################
# Gradient Boosting
####################################################
for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")
        # Dataset preparation
        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise)
        perm = np.random.permutation(len(X))
        X, y = X.to_numpy()[perm], y.to_numpy()[perm]
        if (len(X.shape) == 1) or (X.shape[1] == 1):
                X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

        err_cln = np.zeros((len(snr_db_vec), len(ensemble_size), KFold_n_splits))
        err_nr, err_r = np.zeros_like(err_cln), np.zeros_like(err_cln)

        for _m_idx, _m in enumerate(ensemble_size):  # iterate number of trees
                print("T=" + str(_m) + " regressors")

                kfold_idx = 0
                for train_index, test_index in kf.split(X):
                        print("\nTRAIN:", train_index, "\nTEST:", test_index)
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        # Plotting all the points
                        # if X_train.shape[1] == 1 and plot_flag:
                        #     plt.figure(figsize=(12, 8))
                        #     plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                        #     plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

                        # - - - CLEAN GRADIENT BOOSTING - - -
                        # Initiating the tree
                        if criterion=="mse":
                                rgb_cln = rGradBoost_MSE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                             min_sample_leaf=min_sample_leaf,
                                                             TrainNoiseCov=np.zeros([_m + 1, _m + 1]))
                        elif criterion=="mae":
                                rgb_cln = rGradBoost_MAE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                         min_sample_leaf=min_sample_leaf,
                                                         TrainNoiseCov=np.zeros([_m + 1, _m + 1]))

                        # Fitting on training data
                        rgb_cln.fit(X_train, y_train, m=_m)

                        # Predicting without noise (for reference)
                        pred_cln = rgb_cln.predict(X_test, PredNoiseCov=np.zeros([_m + 1, _m + 1]))
                        # Saving the predictions to the training set
                        err_cln[:, _m_idx, kfold_idx] = np.abs(np.subtract(y_test[:, 0], pred_cln)).mean()
                        # - - - - - - - - - - - - - - - - -

                        for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                                # Set noise variance
                                snr = 10 ** (snr_db / 10)
                                sig_var = np.var(y_train)

                                # Setting noise covariance matrix
                                if _m == 0:  # predictor is the mean of training set
                                        sigma_profile = sig_var / snr  # noise variance
                                        noise_covariance = np.diag(np.reshape(sigma_profile, (1,)))
                                elif sigma_profile_type == "uniform":
                                        sigma0 = sig_var / (snr * _m)
                                        noise_covariance = np.diag(sigma0 * np.ones([_m + 1, ]))
                                elif sigma_profile_type == "good-outlier":
                                        a = 1 / 10  # outlier variance ratio
                                        sigma0 = sig_var / (snr * (a + _m - 1))
                                        tmp = sigma0 * np.ones([_m + 1, ])
                                        tmp[0] *= a
                                        noise_covariance = np.diag(tmp)
                                elif sigma_profile_type == "linear":
                                        sigma_profile = np.linspace(1, 1 / (_m + 1), _m + 1)
                                        curr_snr = sig_var / sigma_profile.sum()
                                        sigma_profile *= curr_snr / snr
                                        noise_covariance = np.diag(sigma_profile)

                                # - - - NON-ROBUST / ROBUST GRADIENT BOOSTING - - -
                                if criterion == "mse":
                                        rgb_nr = rGradBoost_MSE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                 min_sample_leaf=min_sample_leaf,
                                                                 TrainNoiseCov=np.zeros([_m + 1, _m + 1]))
                                        rgb_r = rGradBoost_MSE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                min_sample_leaf=min_sample_leaf,
                                                                TrainNoiseCov=noise_covariance)
                                elif criterion == "mae":
                                        rgb_nr = rGradBoost_MAE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                 min_sample_leaf=min_sample_leaf,
                                                                 TrainNoiseCov=np.zeros([_m + 1, _m + 1]))
                                        rgb_r = rGradBoost_MAE(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                 min_sample_leaf=min_sample_leaf,
                                                                 TrainNoiseCov=noise_covariance)

                                # Fitting on training data with noise: non-robust and robust
                                rgb_nr.fit(X_train, y_train, m=_m)
                                rgb_r.fit(X_train, y_train, m=_m)
                                # - - - - - - - - - - - - - - - - -

                                # Predicting with noise (for reference)
                                pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                for _n in range(0, n_repeat):
                                        # - - - NON-ROBUST - - -
                                        pred_nr = rgb_nr.predict(X_test, PredNoiseCov=noise_covariance)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] += np.abs(
                                                np.subtract(y_test[:, 0], pred_nr)).mean()

                                        # - - - ROBUST - - -
                                        pred_r = rgb_r.predict(X_test, PredNoiseCov=noise_covariance)
                                        err_r[idx_snr_db, _m_idx, kfold_idx] += np.abs(
                                                np.subtract(y_test[:, 0], pred_r)).mean()

                                # Expectation of error (over multiple realizations)
                                err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                print("Error [dB], (Clean, Non-robust, Robust) = (" +
                                      "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                      "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                      "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                                      )

                                # Sample presentation of data
                                if X_train.shape[1] == 1 and plot_flag:
                                        fig_dataset = plt.figure(figsize=(12, 8))
                                        plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                                        plt.plot(X_test[:, 0], pred_cln, 'o',
                                                 label="Clean, " + "err=" + "{:.4f}".format(
                                                         err_cln[0, _m_idx, kfold_idx]))
                                        plt.plot(X_test[:, 0], pred_r, 'o',
                                                 label="Robust, " + "err=" + "{:.4f}".format(
                                                         err_r[idx_snr_db, _m_idx, kfold_idx]))
                                        plt.plot(X_test[:, 0], pred_nr, 'd',
                                                 label="Non-Robust, " + "err=" + "{:.4f}".format(
                                                         err_nr[idx_snr_db, _m_idx, kfold_idx]))
                                        plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
                                        plt.xlabel('x')
                                        plt.ylabel('y')
                                        plt.legend()
                                        plt.show(block=False)
                                        plt.close(fig_dataset)

                        kfold_idx += 1

                if save_results_to_file_flag:
                        results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                                'GradBoost, Noiseless': pd.Series(np.log10(err_cln[:, _m_idx, :].mean(1))),
                                                'GradBoost, Non-Robust': pd.Series(np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                'GradBoost, Robust': pd.Series(np.log10(err_r[:, _m_idx, :].mean(1)))},
                                               axis=1)
                        results_df.to_csv(results_path + data_type + "_" + _m.__str__() + "_gbr_lae.csv")
                print("---------------------------------------------------------------------------\n")

        # Plot error and error gain
        for _m_idx, _m in enumerate(ensemble_size):
                plt.figure(figsize=(12, 8))
                plt.plot(snr_db_vec, 10 * np.log10(err_cln[:, _m_idx, :].mean(1)), '-k', label='Clean')
                plt.plot(snr_db_vec, 10 * np.log10(err_nr[:, _m_idx, :].mean(1)), '-xr', label='Non-robust')
                plt.plot(snr_db_vec, 10 * np.log10(err_r[:, _m_idx, :].mean(1)), '-ob', label='Robust')
                plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                plt.xlabel('SNR [dB]')
                plt.ylabel('LAE [dB]')
                plt.legend()
                plt.show(block=False)

                plt.figure(figsize=(12, 8))
                plt.plot(snr_db_vec,
                         10 * np.log10(err_nr[:, _m_idx, :].mean(1)) - 10 * np.log10(err_r[:, _m_idx, :].mean(1)),
                         '-ob', label='Robust')
                plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                plt.xlabel('SNR [dB]')
                plt.ylabel('LAE Gain [dB]')
                plt.show(block=False)

####################################################
# Bagging
####################################################
for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")
        # Dataset preparation
        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise)
        perm = np.random.permutation(len(X))
        X, y = X.to_numpy()[perm], y.to_numpy()[perm]
        if (len(X.shape) == 1) or (X.shape[1] == 1):
                X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

        mse_results_bem, mse_results_gem, mse_results_lr = [], [], []

        for _m_idx, _m in enumerate(ensemble_size):
                print("T=" + str(_m) + " regressors")

                kfold_idx = 0
                for train_index, test_index in kf.split(X):
                        print("\nTRAIN:", train_index, "\nTEST:", test_index)
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        # Plotting all the points
                        # if X_train.shape[1] == 1 and plot_flag:
                        #     plt.figure(figsize=(12, 8))
                        #     plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                        #     plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

                        # - - - CLEAN BAGGING - - -
                        # Initiating the tree
                        if criterion=="mse":
                                cln_bem = sklearn.ensemble.BaggingRegressor(
                                        sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                                        n_estimators=_m, random_state=rng)
                        elif criterion=="mae":
                                cln_bem = sklearn.ensemble.BaggingRegressor(
                                        sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth,
                                                                      criterion="absolute_error"),
                                        n_estimators=_m, random_state=rng)

                        # Fitting on training data
                        cln_bem.fit(X_train, y_train)

                        # Predicting without noise (for reference)
                        pred_cln = cln_bem.predict(X_test)
                        # Saving the predictions to the training set
                        err_cln[:, _m_idx, kfold_idx] = aux.calc_error(y_test[:, 0], pred_cln, criterion)
                        # - - - - - - - - - - - - - - - - -

                        for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                                # Set noise variance
                                snr = 10 ** (snr_db / 10)
                                sig_var = np.var(y_train)

                                # Setting noise covariance matrix
                                if sigma_profile_type == "uniform":
                                        sigma0 = sig_var / (snr * _m)
                                        noise_covariance = np.diag(sigma0 * np.ones([_m + 1, ]))
                                elif sigma_profile_type == "good-outlier":
                                        a = 1 / 10  # outlier variance ratio
                                        sigma0 = sig_var / (snr * (a + _m - 1))
                                        tmp = sigma0 * np.ones([_m + 1, ])
                                        tmp[0] *= a
                                        noise_covariance = np.diag(tmp)
                                elif sigma_profile_type == "linear":
                                        sigma_profile = np.linspace(1, 1 / (_m + 1), _m + 1)
                                        curr_snr = sig_var / sigma_profile.sum()
                                        sigma_profile *= curr_snr / snr
                                        noise_covariance = np.diag(sigma_profile)

                                # - - - NON-ROBUST / ROBUST GRADIENT BOOSTING - - -
                                if criterion == "mse":
                                        noisy_bem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'bem')
                                        noisy_rbem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'robust-bem')
                                        noisy_gem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'gem')
                                        noisy_rgem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'robust-gem')
                                        noisy_lr = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'lr')
                                        noisy_rlr = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'robust-lr')
                                elif criterion == "mae":
                                        noisy_bem = rBaggReg_MAE(cln_bem, noise_covariance, _m, 'bem')
                                        noisy_rbem = rBaggReg_MAE(cln_bem, noise_covariance, _m, 'robust-bem')
                                        noisy_gem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'gem')
                                        noisy_rgem = rBaggReg_MSE(cln_bem, noise_covariance, _m, 'robust-gem')

                                # Fitting on training data with noise: non-robust and robust
                                if criterion == "mse":
                                        noisy_bem.fit(X_train, y_train)
                                        noisy_rbem.fit(X_train, y_train)
                                        noisy_gem.fit(X_train, y_train)
                                        noisy_rgem.fit(X_train, y_train)
                                        noisy_lr.fit(X_train, y_train)
                                        noisy_rlr.fit(X_train, y_train)
                                elif criterion == "mae":
                                        noisy_bem.fit(X_train, y_train)
                                        noisy_rbem.fit(X_train, y_train)
                                        noisy_gem.fit(X_train, y_train)
                                        noisy_rgem.fit(X_train, y_train)
                                # - - - - - - - - - - - - - - - - -

                                # ======== REACHED HERE...
                                # Predicting with noise (for reference)
                                pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                for _n in range(0, n_repeat):
                                        # - - - NON-ROBUST - - -
                                        pred_nr = rgb_nr.predict(X_test, PredNoiseCov=noise_covariance)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] += np.abs(
                                                np.subtract(y_test[:, 0], pred_nr)).mean()

                                        # - - - ROBUST - - -
                                        pred_r = rgb_r.predict(X_test, PredNoiseCov=noise_covariance)
                                        err_r[idx_snr_db, _m_idx, kfold_idx] += np.abs(
                                                np.subtract(y_test[:, 0], pred_r)).mean()

                                # Expectation of error (over multiple realizations)
                                err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                print("Error [dB], (Clean, Non-robust, Robust) = (" +
                                      "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                      "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                      "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                                      )

                                # Sample presentation of data
                                if X_train.shape[1] == 1 and plot_flag:
                                        fig_dataset = plt.figure(figsize=(12, 8))
                                        plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                                        plt.plot(X_test[:, 0], pred_cln, 'o',
                                                 label="Clean, " + "err=" + "{:.4f}".format(
                                                         err_cln[0, _m_idx, kfold_idx]))
                                        plt.plot(X_test[:, 0], pred_r, 'o',
                                                 label="Robust, " + "err=" + "{:.4f}".format(
                                                         err_r[idx_snr_db, _m_idx, kfold_idx]))
                                        plt.plot(X_test[:, 0], pred_nr, 'd',
                                                 label="Non-Robust, " + "err=" + "{:.4f}".format(
                                                         err_nr[idx_snr_db, _m_idx, kfold_idx]))
                                        plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
                                        plt.xlabel('x')
                                        plt.ylabel('y')
                                        plt.legend()
                                        plt.show(block=False)
                                        plt.close(fig_dataset)

                        kfold_idx += 1

                if save_results_to_file_flag:
                        results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                                'GradBoost, Noiseless': pd.Series(np.log10(err_cln[:, _m_idx, :].mean(1))),
                                                'GradBoost, Non-Robust': pd.Series(np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                'GradBoost, Robust': pd.Series(np.log10(err_r[:, _m_idx, :].mean(1)))},
                                               axis=1)
                        results_df.to_csv(results_path + data_type + "_" + _m.__str__() + "_gbr_lae.csv")
                print("---------------------------------------------------------------------------\n")


















============================================================================================
                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                        snr = 10**(snr_db/10)

                        # Get data set
                        X_train, y_train, X_test, y_test = aux.partition_dataset(data_type=data_type, test_size=test_size, n_samples=n_samples, noise=train_noise)
                        X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

                        # Set prediction noise
                        sig_var = np.var(y_train)
                        sigma0 = sig_var/snr  # noise variance

                        sigma_profile = np.ones([n_estimators, 1])
                        # sigma_profile[1:n_estimators-1, :] = 0 #sigma_profile[1:9, :] / 100
                        # sigma_profile[5:n_estimators-1, :] = 0 #sigma_profile[1::2, :] / 100
                        # sigma_profile[n_estimators-1, :] = 0 #sigma_profile[1::2, :] / 100
                        # sigma_profile[1:n_estimators-1, :] = sigma_profile[1:n_estimators-1, :] / 100
                        sigma_profile[5:n_estimators-1, :] = sigma_profile[5:n_estimators-1, :] / 100
                        # sigma_profile[n_estimators-1, :] = sigma_profile[n_estimators-1, :] / 100
                        sigma_profile /= sigma_profile.sum()
                        sigma_profile *= sigma0
                        noise_covariance = np.diag(sigma_profile.ravel())

                        # - - - BEM
                        # Fit regression model
                        regr_1_bem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                        regr_2_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'bem')
                        regr_3_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'robust-bem')

                        regr_1_bem.fit(X_train, y_train)
                        regr_2_bem.fit(X_train, y_train)
                        regr_3_bem.fit(X_train, y_train)

                        # Predict
                        y_1 = regr_1_bem.predict(X_test)
                        y_2 = regr_2_bem.predict(X_test)
                        y_3 = regr_3_bem.predict(X_test)

                        # ind_train = np.argsort(X_train[:, 0])
                        # ind_test = np.argsort(X_test[:, 0])
                        # plt.figure()
                        # plt.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='', marker='o',
                        #          markersize=2)
                        # plt.plot(X_test[ind_test], y_2[ind_test], c="b", label="Prior", linestyle='', linewidth=2, Marker='.',
                        #          markersize=2)
                        # plt.plot(X_test[ind_test], y_3[ind_test], c="g", label="Robust", linestyle='-', linewidth=2)
                        # plt.xlabel("Data")
                        # plt.ylabel("Target")
                        # plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
                        # plt.legend()
                        # plt.show(block=False)

                        if plot_flag:
                                # Plot model accuracy (no noise)
                                plt.figure()
                                plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                                plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='o', markersize=1)

                                # Plot the results
                                if X_test.shape[1] <= 1:
                                        ind_train = np.argsort(X_train[:,0])
                                        ind_test = np.argsort(X_test[:,0])
                                        plt.figure()
                                        # plt.plot(X_test[ind], y_test[ind], c="k", label="Target", linestyle='-', linewidth=1)
                                        plt.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='-', marker='o', markersize=1)
                                        # plt.plot(X_test[ind], y_1[ind], c="r", label="Clean", linestyle='--', linewidth=2)
                                        plt.plot(X_test[ind_test], y_2[ind_test], c="b", label="Prior", linestyle='', linewidth=2, Marker='.', markersize=1.5)
                                        plt.plot(X_test[ind_test], y_3[ind_test], c="g", label="Robust", linestyle=':', linewidth=2)
                                        plt.xlabel("Data")
                                        plt.ylabel("Target")
                                        plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
                                        plt.legend()
                                        plt.show(block=False)

                                plt.figure()
                                plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                                # plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='x', markersize=3)
                                plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                                plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                                plt.xlabel('Test target')
                                plt.ylabel('Test prediction')
                                plt.title('BEM')
                                plt.grid()
                                plt.legend()

                        print("BEM MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                        "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                        "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                              )
                        mse_results_bem.append([ snr_db,
                                                sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

                        # - - - GEM
                        # Fit regression model
                        regr_1_gem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                        regr_2_gem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_gem, noise_covariance, n_estimators, 'gem')
                        regr_3_gem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_gem, noise_covariance, n_estimators, 'robust-gem')

                        regr_1_gem.fit(X_train, y_train)
                        regr_2_gem.fit(X_train, y_train)
                        regr_3_gem.fit(X_train, y_train)

                        # Predict
                        y_1 = regr_1_gem.predict(X_test)
                        y_2 = regr_2_gem.predict(X_test)
                        y_3 = regr_3_gem.predict(X_test)

                        # Plot the results
                        if plot_flag:
                                plt.figure()
                                plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                                plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='.', markersize=2)
                                plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                                plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                                plt.xlabel('Test target')
                                plt.ylabel('Test prediction')
                                plt.title('GEM')
                                plt.grid()
                                plt.legend()

                        print("GEM MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                        "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                        "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                              )
                        mse_results_gem.append([ snr_db,
                                                sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

                        # - - - LR
                        # Fit regression model
                        regr_1_lr = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                        regr_2_lr = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_lr, noise_covariance, n_estimators, 'lr')
                        regr_3_lr = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_lr, noise_covariance, n_estimators, 'robust-lr')

                        regr_1_lr.fit(X_train, y_train)
                        regr_2_lr.fit(X_train, y_train)
                        regr_3_lr.fit(X_train, y_train)

                        # Predict
                        y_1 = regr_1_lr.predict(X_test)
                        y_2 = regr_2_lr.predict(X_test)
                        y_3 = regr_3_lr.predict(X_test)

                        # Plot the results
                        if plot_flag:
                                plt.figure()
                                plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                                plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='.', markersize=2)
                                plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                                plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                                plt.xlabel('Test target')
                                plt.ylabel('Test prediction')
                                plt.title('LR')
                                plt.grid()
                                plt.legend()

                        print("LR MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                        "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                        "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                              )
                        mse_results_lr.append([ snr_db,
                                                sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                                sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

                mse_results_bem_df = pd.DataFrame(mse_results_bem, columns=['SNR', 'BEM, Noiseless', 'BEM, Prior', 'BEM, Robust'])
                mse_results_gem_df = pd.DataFrame(mse_results_gem, columns=['SNR', 'GEM, Noiseless', 'GEM, Prior', 'GEM, Robust'])
                mse_results_lr_df = pd.DataFrame(mse_results_lr, columns=['SNR', 'LR, Noiseless', 'LR, Prior', 'LR, Robust'])

                mse_results_bem_df.to_csv(results_path+data_type+"_bem.csv")
                mse_results_gem_df.to_csv(results_path+data_type+"_gem.csv")
                mse_results_lr_df.to_csv(results_path+data_type+"_lr.csv")

if plot_flag:
        plt.figure()
        plt.plot(snr_db_vec, 10*np.log10(mse_results_bem_df['BEM, Prior']/mse_results_bem_df['BEM, Noiseless']), c="k", label="BEM", linestyle='-', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_bem_df['BEM, Robust']/mse_results_bem_df['BEM, Noiseless']), c="k", label="rBEM", linestyle='-', marker='*', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_gem_df['GEM, Prior']/mse_results_gem_df['GEM, Noiseless']), c="r", label="GEM", linestyle='--', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_gem_df['GEM, Robust']/mse_results_gem_df['GEM, Noiseless']), c="r", label="rGEM", linestyle='--', marker='*', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_lr_df['LR, Prior']/mse_results_lr_df['LR, Noiseless']), c="b", label="LR", linestyle='-.', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_lr_df['LR, Robust']/mse_results_lr_df['LR, Noiseless']), c="b", label="rLR", linestyle='-.', marker='*', linewidth=0.75)
        plt.xlabel('SNR [dB]')
        plt.ylabel('Relative MSE (Test) [dB]')
        # plt.title('')
        plt.grid()
        plt.legend()
