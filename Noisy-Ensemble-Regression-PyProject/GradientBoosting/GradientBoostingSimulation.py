import numpy as np
import matplotlib.pyplot as plt
# OS traversal
import os
# System functionalities
import sys
# Adding the whole project to module paths
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:

sys.path.append(module_path)
import pandas as pd
import matplotlib.pyplot as plt  # Plotting
from GradientBoosting.boosting import RobustRegressionGB  # Regression boosting
import RobustIntegration.auxilliaryFunctions as aux
from sklearn.model_selection import KFold  # k-fold cross-validation

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Constants
rng = np.random.RandomState(42)
plt.rcParams.update({'font.size': 20})
cm = 1/2.54  # [inch]
fsiz = 20*cm  # figure size [inch]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings
plot_flag = True
nmse_or_mse = "mse"
results_path = "Results//"

# Parameters
data_type_vec = ["diabetes"]  # ["auto-mpg", "kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]
snr_db_vec = np.linspace(-25, 15, 5)
sigma_profile_type = "linear"  # uniform / good-outlier / linear
_m_iterations = [1, 32, 128]  # Number of weak-learners

KFold_n_splits = 2  # Number of k-fold x-validation dataset splits
n_repeat = 500  # Number of iterations for estimating expected performance

n_samples, test_size = 500, 0.2 # Size of the (synthetic) dataset / test data fraction
train_noise = 0.1  # Standard deviation of the measurement / training noise
max_depth = 1  # Maximal depth of decision tree
learning_rate = 0.1  # learning rate of gradient boosting
min_sample_leaf = 10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

####################################################
# Gradient Boosting
####################################################
if True:
    for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")

        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise)
        perm = np.random.permutation(len(X))
        X, y = X.to_numpy()[perm], y.to_numpy()[perm]
        if (len(X.shape) == 1) or (X.shape[1] == 1):
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

        mse_cln = np.zeros((len(snr_db_vec), len(_m_iterations), KFold_n_splits))
        mse_nr, mse_r = np.zeros_like(mse_cln), np.zeros_like(mse_cln)

        for _m_idx, _m in enumerate(_m_iterations):  # iterate number of trees
            print("T=" + str(_m) + " regressors")

            kfold_idx = 0
            for train_index, test_index in kf.split(X):
                print("\nTRAIN:", train_index, "\nTEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # - - - CLEAN GRADIENT BOOSTING - - -
                # Initiating the tree
                rgb_cln = RobustRegressionGB(X=X_train, y=y_train, max_depth=max_depth, min_sample_leaf=min_sample_leaf,
                                            NoiseCov=np.zeros([_m+1,_m+1]), RobustFlag=0)
                # Fitting on data
                rgb_cln.fit(X_train, y_train, m=_m)
                # Predicting
                y_test_hat = rgb_cln.predict(X_test)
                # Saving the predictions to the training set
                pred_cln = y_test_hat
                mse_cln[:, _m_idx, kfold_idx] = np.sqrt(np.square(np.subtract(y_test[:, 0], y_test_hat)).mean())
                if nmse_or_mse == "nmse":
                    mse_cln[:, _m_idx, kfold_idx] /= np.square(y_test[:, 0]).mean()
                # - - - - - - - - - - - - - - - - -

                # Plotting dataset: Training data, Test data, "clean" predictions
                if X_train.shape[1] == 1 and plot_flag:
                    fig_dataset = plt.figure(figsize=(12, 8))
                    plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                    plt.plot(X_test[:, 0], y_test[:, 0], 'xr', label='Test')
                    plt.plot(X_test[:, 0], y_test_hat, 'or', label='Prediction (Test)')
                    plt.title("_m=" + "{:d}".format(_m) + ", mse=" + "{:.4f}".format(mse_cln[0, _m_idx, kfold_idx]))
                    plt.legend()
                    plt.show(block=False)
                    plt.pause(0.05)
                    plt.close(fig_dataset)

                if True:
                    for idx_snr_db, snr_db in enumerate(snr_db_vec):
                        print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end =" ")

                        # Set noise variance
                        snr = 10 ** (snr_db / 10)
                        sig_var = np.var(y_train)

                        # Setting noise covariance matrix
                        if _m == 0:  # predictor is the mean of training set
                            sigma_profile = sig_var / snr  # noise variance
                            noise_covariance = np.diag(np.reshape(sigma_profile,(1,)))
                        elif sigma_profile_type == "uniform":
                            sigma0 = sig_var / (snr * _m)
                            noise_covariance = np.diag(sigma0 * np.ones([_m + 1, ]))
                        elif sigma_profile_type == "good-outlier":
                            a = 1/10  # outlier variance ratio
                            sigma0 = sig_var / (snr * (a+_m-1))
                            tmp = sigma0 * np.ones([_m + 1, ])
                            tmp[0] *= a
                            noise_covariance = np.diag(tmp)
                        elif sigma_profile_type == "linear":
                            sigma_profile = np.linspace(1, 1/(_m+1), _m+1)
                            curr_snr = sig_var / sigma_profile.sum()
                            sigma_profile *= curr_snr/snr
                            noise_covariance = np.diag(sigma_profile)


                        # - - - NON-ROBUST GRADIENT BOOSTING - - -
                        # Initiating the tree
                        rgb_nr = RobustRegressionGB(X=X_train, y=y_train, max_depth=max_depth, min_sample_leaf=min_sample_leaf,
                                                NoiseCov=noise_covariance, RobustFlag=0)
                        # Fitting on data
                        rgb_nr.fit(X_train, y_train, m=_m)
                        # - - - - - - - - - - - - - - - - -

                        # - - - ROBUST GRADIENT BOOSTING - - -
                        rgb_r = RobustRegressionGB(X=X_train, y=y_train, max_depth=max_depth, min_sample_leaf=min_sample_leaf,
                                                NoiseCov=noise_covariance, RobustFlag=1)
                        # Fitting on data
                        rgb_r.fit(X_train, y_train, m=_m)
                        # - - - - - - - - - - - - - - - - -

                        pred_nr = np.zeros(len(y_test))
                        pred_r = np.zeros(len(y_test))
                        for _n in range(0, n_repeat):
                            # - - - NON-ROBUST - - -
                            # Predicting
                            y_test_hat = rgb_nr.predict(X_test)
                            # Saving the predictions to the training set
                            mse_nr[idx_snr_db, _m_idx, kfold_idx] += np.sqrt(np.square(np.subtract(y_test[:, 0], y_test_hat)).mean())
                            if nmse_or_mse == "nmse":
                                mse_nr[idx_snr_db, _m_idx, kfold_idx] /= np.square(y_test[:, 0]).mean()
                            pred_nr = y_test_hat
                            # - - - - - - - - - - - - - - - - -

                            # - - - ROBUST - - -
                            # Predicting
                            y_test_hat = rgb_r.predict(X_test)
                            # Saving the predictions to the training set
                            mse_r[idx_snr_db, _m_idx, kfold_idx] += np.sqrt(np.square(np.subtract(y_test[:,0], y_test_hat)).mean())
                            if nmse_or_mse == "nmse":
                                mse_r[idx_snr_db, _m_idx, kfold_idx] /= np.square(y_test[:, 0]).mean()
                            pred_r = y_test_hat
                            # - - - - - - - - - - - - - - - - -

                        mse_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                        mse_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                        print("MSE, (Clean, Non-robust, Robust) = (" +
                              "{0:0.3f}".format(10 * np.log10(mse_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                              "{0:0.3f}".format(10 * np.log10(mse_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                              "{0:0.3f}".format(10 * np.log10(mse_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                              )

                        # Plotting dataset: Training data, Test data, "clean" predictions
                        if X_train.shape[1]==1 and plot_flag:
                            fig_dataset = plt.figure(figsize=(12, 8))
                            plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                            plt.plot(X_test[:, 0], pred_cln, 'o', label="Clean, " + "mse=" + "{:.4f}".format(mse_cln[0, _m_idx, kfold_idx]))
                            plt.plot(X_test[:, 0], pred_r, 'o', label="Robust, " + "mse=" + "{:.4f}".format(mse_r[idx_snr_db, _m_idx, kfold_idx]))
                            plt.plot(X_test[:, 0], pred_nr, 'd', label="Non-Robust, " + "mse=" + "{:.4f}".format(mse_nr[idx_snr_db, _m_idx, kfold_idx]))
                            plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
                            plt.xlabel('x')
                            plt.ylabel('y')
                            plt.legend()
                            plt.show(block=False)
                            plt.draw()
                            plt.close(fig_dataset)

                kfold_idx += 1

            results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                   'GBR, Noiseless': pd.Series(np.log10(mse_cln[:, _m_idx, :].mean(1))),
                                    'GBR, Non-Robust': pd.Series(np.log10(mse_nr[:, _m_idx, :].mean(1))),
                                    'GBR, Robust': pd.Series(np.log10(mse_r[:, _m_idx, :].mean(1)))},
                                   axis=1)
            results_df.to_csv(results_path + data_type + "_" + _m.__str__() + "_gbr.csv")
            print("---------------------------------------------------------------------------\n")

        # Plot MSE results
        for _m_idx, _m in enumerate(_m_iterations):
            plt.figure(figsize=(12, 8))
            plt.plot(snr_db_vec, 10 * np.log10(mse_cln[:, _m_idx, :].mean(1)), '-k', label='Clean')
            plt.plot(snr_db_vec, 10 * np.log10(mse_nr[:, _m_idx, :].mean(1)), '-xr', label='Non-robust')
            plt.plot(snr_db_vec, 10 * np.log10(mse_r[:, _m_idx, :].mean(1)), '-ob', label='Robust')
            plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
            plt.xlabel('SNR [dB]')
            plt.ylabel('MSE [dB]')
            if nmse_or_mse == "nmse":
                plt.ylabel('NMSE [dB]')
            plt.legend()
            plt.show(block=False)

        # Plot MSE gain results
        # for _m_idx, _m in enumerate(_m_iterations):
        #     plt.figure(figsize=(12, 8))
        #     plt.plot(snr_db_vec, 10 * np.log10(mse_nr[:, _m_idx, :].mean(1)) - 10 * np.log10(mse_cln[:, _m_idx, :].mean(1)), '-xr', label='Non-robust')
        #     plt.plot(snr_db_vec, 10 * np.log10(mse_r[:, _m_idx, :].mean(1)) - 10 * np.log10(mse_cln[:, _m_idx, :].mean(1)), '-ob', label='Robust')
        #     plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
        #     plt.xlabel('SNR [dB]')
        #     plt.ylabel('MSE Gain [dB]')
        #     if nmse_or_mse == "nmse":
        #         plt.ylabel('NMSE [dB]')
        #     plt.legend()
        #     plt.show(block=False)
