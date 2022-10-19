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
# Data wrangling
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
# Regression boosting
from GradientBoosting.boosting import RobustRegressionGB

import RobustIntegration.auxilliaryFunctions as aux
import sklearn as sk
from sklearn.model_selection import KFold

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rng = np.random.RandomState(42)
plt.rcParams.update({'font.size': 20})
cm = 1/2.54
fsiz = 20*cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings
plot_flag = False
nmse_or_mse = "nmse"
results_path = "Results//"

n_repeat = 500  # Number of iterations for estimating expected performance
n_samples, test_size = 500, 0.2 # Size of the (synthetic) dataset / test data fraction
train_noise = 0.1  # Standard deviation of the measurement / training noise
max_depth = 1  # Maximal depth of decision tree
learning_rate = 0.1  # learning rate of gradient boosting
snr_db_vec = np.linspace(-25, 15, 15)

# xv, yv = np.meshgrid(x_name, y_name, sparse=False, indexing='ij')
# noise_covariance = np.exp(xv-yv)

####################################################
# Gradient Boosting
####################################################
if True:
    data_type_vec = ["auto-mpg"]  # ["auto-mpg", "kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]
    sigma_profile_type = "uniform"  # uniform / good-outlier / half
    for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")

        # Defining (possibly dataset dependent) parameters
        _m_iterations = [1, 4]  # Number of weak-learners
        KFold_n_splits = 10  # Number of k-fold x-validation dataset splits

        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise)
        X, y = X.to_numpy(), y.to_numpy()
        if (len(X.shape) == 1) or (X.shape[1] == 1):
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        mse    = np.zeros((len(snr_db_vec), len(_m_iterations), KFold_n_splits))
        mse_nr = np.zeros_like(mse)

        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)
        kfold_idx = 0
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "\nTEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for _m_idx, _m in enumerate(_m_iterations):  # iterate number of trees
                print("T=" + str(_m) + " regressors")

                # Plotting all the points
                if X_train.shape[1] == 1 and plot_flag:
                    plt.figure(figsize=(12, 8))
                    plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                    plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                    print(" - snr " + " = " + str(snr_db) + " - ")

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
                    elif sigma_profile_type == "half":
                        # TODO
                        tmp=1

                    # - - - ROBUST GRADIENT BOOSTING - - -
                    rgb = RobustRegressionGB(
                        X=X_train,
                        y=y_train,
                        max_depth=max_depth,
                        min_sample_leaf=10,
                        learning_rate=learning_rate,
                        NoiseCov=noise_covariance,
                        RobustFlag=1
                    )
                    # Fitting on data
                    rgb.fit(X_train, y_train, m=_m)

                    # - - - NON-ROBUST GRADIENT BOOSTING - - -
                    # Initiating the tree
                    rgb_nr = RobustRegressionGB(
                        X=X_train,
                        y=y_train,
                        max_depth=max_depth,
                        min_sample_leaf=10,
                        learning_rate=learning_rate,
                        NoiseCov=noise_covariance,
                        RobustFlag=0
                    )
                    # Fitting on data
                    rgb_nr.fit(X_train, y_train, m=_m)

                    pred = np.zeros(len(y_test))
                    pred_nr = np.zeros(len(y_test))
                    for _n in range(0, n_repeat):
                        # ROBUST
                        # Predicting
                        y_test_hat = rgb.predict(X_test)
                        # Saving the predictions to the training set
                        mse[idx_snr_db, _m_idx, kfold_idx] += np.square(np.subtract(y_test[:,0], y_test_hat)).mean()
                        if nmse_or_mse == "nmse":
                            # mse[idx_snr_db, _m_idx, kfold_idx] /= np.sqrt(np.square(y_test[:, 0]).mean() * np.square(y_test_hat).mean())
                            mse[idx_snr_db, _m_idx, kfold_idx] /= np.square(y_test[:, 0]).mean()
                        pred += y_test_hat

                        # NON-ROBUST
                        # Predicting
                        y_test_hat = rgb_nr.predict(X_test)
                        # Saving the predictions to the training set
                        mse_nr[idx_snr_db, _m_idx, kfold_idx] += np.square(np.subtract(y_test[:, 0], y_test_hat)).mean()
                        if nmse_or_mse == "nmse":
                            # mse_nr[idx_snr_db, _m_idx, kfold_idx] /= np.sqrt(np.square(y_test[:, 0]).mean() * np.square(y_test_hat).mean())
                            mse_nr[idx_snr_db, _m_idx, kfold_idx] /= np.square(y_test[:, 0]).mean()
                        pred_nr += y_test_hat

                    mse[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                    pred /= n_repeat
                    mse_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                    pred_nr /= n_repeat

                    print("\tNon-robust mse="+str(10*np.log10(mse_nr[idx_snr_db, _m_idx, kfold_idx])))
                    print("\tRobust mse=" + str(10*np.log10(mse[idx_snr_db, _m_idx, kfold_idx])))

                    if X_train.shape[1]==1 and plot_flag:
                        plt.plot(X_test[:,0], pred, 'o', label=f't={_m}, mse={mse[idx_snr_db]}')
                        plt.plot(X_test[:,0], pred_nr, 'd', label=f't={_m}, mse={mse_nr[idx_snr_db]}')
                        plt.title('mpg vs weight')
                        plt.xlabel('weight')
                        plt.ylabel('mpg')
                        plt.legend()
                        plt.show()

            kfold_idx += 1
            print("\n---------------------------------------------------------------------------\n")

        for _m_idx, _m in enumerate(_m_iterations):
            plt.figure(figsize=(12, 8))
            plt.plot(snr_db_vec, 10 * np.log10(mse_nr[:, _m_idx, :].mean(1)), '-xk', label='Non-robust')
            plt.plot(snr_db_vec, 10 * np.log10(mse[:, _m_idx, :].mean(1)), '-ok', label='Robust')
            plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
            plt.xlabel('SNR [dB]')
            plt.ylabel('MSE [dB]')
            if nmse_or_mse == "nmse":
                plt.ylabel('NMSE [dB]')
            plt.legend()
            plt.show()
