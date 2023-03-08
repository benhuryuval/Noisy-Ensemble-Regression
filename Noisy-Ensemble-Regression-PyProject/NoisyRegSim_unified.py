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
from robustRegressors.rBaggReg import rBaggReg as rBaggReg
from robustRegressors.rGradBoost import rGradBoost as rGradBoost
import RobustIntegration.auxilliaryFunctions as aux
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constants
rng = np.random.RandomState(42)
plot_flag = False
save_results_to_file_flag = True
results_path = "Results//"

data_type_vec = ["kc_house_data"]  # kc_house_data / diabetes / white-wine / sin / exp / make_reg
# data_type_vec = ["sin", "exp", "make_reg", "diabetes", "white-wine", "kc_house_data"]
KFold_n_splits = 2  # Number of k-fold x-validation dataset splits

ensemble_size = [8] # [16, 64] # [5] # Number of weak-learners
tree_max_depth = 1  # Maximal depth of decision tree
min_sample_leaf = 10

snr_db_vec = np.linspace(-40, 20, 7)  # [-6]
n_repeat = 1  # Number of iterations for estimating expected performance
sigma_profile_type = "noiseless_even"  # uniform / linear / noiseless_fraction / noiseless_even (for GradBoost)
noisless_fraction = 0.25
noisless_scale = 1/100

n_samples = 500  # Size of the (synthetic) dataset  in case of synthetic dataset
train_noise = 0.01  # Standard deviation of the measurement / training noise in case of synthetic dataset

criterion = "mae"  # "mse" / "mae"
reg_algo = "GradBoost"  # "GradBoost" / "Bagging"
bagging_method = "gem"  # "bem" / "gem" / "lr"
gradboost_robust_flag = True

# Gradient-descent params
if reg_algo == "Bagging":
    gd_learn_rate_dict = {
        "sin": 1e-2,
        "exp": 1e-2,
        "make_reg": 1e-6,
        "diabetes": 1e-4,
        "white-wine": 1e-4,
        "kc_house_data": 1e-6
    }
    gd_tol = 1e-2  # or 1e-8
    gd_decay_rate = 0.0  # or 0.2
elif reg_algo == "GradBoost":
    gd_learn_rate_dict = {
        "sin": 25e-1,
        "exp": 1e-1,
        "make_reg": 1e-2,
        "diabetes": 1e-4,
        "white-wine": 25e-1,
        "kc_house_data": 1e-7
    }
    gd_tol = 1e-12  # or 1e-12
    gd_decay_rate = 0.0  # or 0.2


# Verify inputs
if reg_algo == "Bagging" and bagging_method == "lr":
        raise ValueError('Invalid bagging_method.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Main simulation loop(s)
####################################################
# Gradient Boosting
####################################################
if reg_algo == "GradBoost":
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
                                print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]

                                # Plotting all the points
                                if X_train.shape[1] == 1 and plot_flag:
                                    plt.figure(figsize=(12, 8))
                                    plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                                    plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

                                # - - - CLEAN GRADIENT BOOSTING - - -
                                # Initiating the tree
                                rgb_cln = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                        min_sample_leaf=min_sample_leaf,
                                                        TrainNoiseCov=np.zeros([_m + 1, _m + 1]),
                                                        RobustFlag = gradboost_robust_flag,
                                                        criterion=criterion)

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
                                        sig_var = np.var(y_train*ensemble_size)

                                        # Setting noise covariance matrix
                                        if _m == 0:
                                                sigma_profile = sig_var / snr  # noise variance
                                                noise_covariance = np.diag(np.reshape(sigma_profile, (1,)))
                                        elif sigma_profile_type == "uniform":
                                                sigma0 = sig_var / (snr * _m)
                                                noise_covariance = np.diag(sigma0 * np.ones([_m + 1, ]))
                                        elif sigma_profile_type == "linear":
                                                sigma_profile = np.linspace(1, 1/(_m+1), _m+1)
                                                sigma0 = sig_var / (snr * sigma_profile.sum())
                                                sigma_profile *= sigma0
                                                noise_covariance = np.diag(sigma_profile)
                                        elif sigma_profile_type == "noiseless_fraction":
                                                sigma0 = sig_var / (snr * (_m+1) * (noisless_fraction + (1-noisless_fraction)/noisless_scale))
                                                sigma_profile = sigma0 * np.ones([_m+1, ])
                                                sigma_profile[0:2:round(noisless_fraction * (_m+1))-1] *= noisless_scale
                                                noise_covariance = np.diag(sigma_profile)
                                        elif sigma_profile_type == "noiseless_even":
                                                sigma0 = sig_var / (snr * (_m+1) * (noisless_fraction + (1-noisless_fraction)/noisless_scale))
                                                sigma_profile = sigma0 * np.ones([_m+1, ])
                                                sigma_profile[0::2] *= noisless_scale
                                                noise_covariance = np.diag(sigma_profile)

                                        # - - - NON-ROBUST / ROBUST GRADIENT BOOSTING - - -
                                        rgb_nr = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                min_sample_leaf=min_sample_leaf,
                                                                TrainNoiseCov=np.zeros([_m + 1, _m + 1]),
                                                                RobustFlag=gradboost_robust_flag,
                                                                criterion=criterion,
                                                                gd_tol=gd_tol, gd_learn_rate=gd_learn_rate_dict[data_type], gd_decay_rate=gd_decay_rate)
                                        rgb_r = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                min_sample_leaf=min_sample_leaf,
                                                                TrainNoiseCov=noise_covariance,
                                                                RobustFlag=gradboost_robust_flag,
                                                                criterion=criterion,
                                                                gd_tol=gd_tol, gd_learn_rate=gd_learn_rate_dict[data_type], gd_decay_rate=gd_decay_rate)

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
                                results_df.to_csv(results_path + data_type + "_gbr_" + _m.__str__() + "_" + criterion + "_" + sigma_profile_type + ".csv")
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
if reg_algo == "Bagging":
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

                for _m_idx, _m in enumerate(ensemble_size):
                        print("T=" + str(_m) + " regressors")

                        err_cln = np.zeros((len(snr_db_vec), len(ensemble_size), KFold_n_splits))
                        err_nr, err_r = np.zeros_like(err_cln), np.zeros_like(err_cln)

                        kfold_idx = -1
                        for train_index, test_index in kf.split(X):
                                print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                kfold_idx += 1

                                # - - - CLEAN BAGGING - - -
                                # Initiating the tree
                                if criterion == "mse":
                                        cln_reg = sklearn.ensemble.BaggingRegressor(
                                                sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                                                n_estimators=_m, random_state=rng)
                                elif criterion == "mae":
                                        cln_reg = sklearn.ensemble.BaggingRegressor(
                                                sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth,
                                                                              criterion="absolute_error"),
                                                n_estimators=_m, random_state=rng)

                                # Fitting on training data
                                cln_reg.fit(X_train, y_train[:, 0])

                                # Predicting without noise (for reference)
                                pred_cln = cln_reg.predict(X_test)
                                # Saving the predictions to the training set
                                err_cln[:, _m_idx, kfold_idx] = aux.calc_error(y_test, pred_cln, criterion)
                                # - - - - - - - - - - - - - - - - -

                                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                        print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                                        # Set noise variance
                                        snr = 10 ** (snr_db / 10)
                                        sig_var = np.var(y_train*_m)

                                        # Setting noise covariance matrix
                                        if sigma_profile_type == "uniform":
                                                sigma0 = sig_var / (snr * _m)
                                                noise_covariance = np.diag(sigma0 * np.ones([_m, ]))
                                        elif sigma_profile_type == "linear":
                                                sigma_profile = np.linspace(1, 1/_m, _m)
                                                sigma0 = sig_var / (snr * sigma_profile.sum())
                                                sigma_profile *= sigma0
                                                noise_covariance = np.diag(sigma_profile)
                                        elif sigma_profile_type == "noiseless_fraction":
                                                sigma0 = sig_var / (snr * _m * (noisless_fraction + (1-noisless_fraction)/noisless_scale))
                                                sigma_profile = sigma0 * np.ones([_m, ])
                                                sigma_profile[0:round(noisless_fraction * _m)-1] *= noisless_scale
                                                noise_covariance = np.diag(sigma_profile)

                                        # - - - NON-ROBUST / ROBUST GRADIENT BOOSTING - - -
                                        noisy_reg = rBaggReg(cln_reg, noise_covariance, _m, bagging_method, gd_tol, gd_learn_rate_dict[data_type], gd_decay_rate)
                                        noisy_rreg = rBaggReg(cln_reg, noise_covariance, _m, "robust-"+bagging_method, gd_tol, gd_learn_rate_dict[data_type], gd_decay_rate)

                                        # Fitting on training data with noise: non-robust and robust
                                        if criterion == "mse":
                                                noisy_reg.fit_mse(X_train, y_train[:, 0])
                                                noisy_rreg.fit_mse(X_train, y_train[:, 0])
                                        elif criterion == "mae":
                                                noisy_reg.fit_mae(X_train, y_train[:, 0])
                                                noisy_rreg.fit_mae(X_train, y_train[:, 0])
                                        # - - - - - - - - - - - - - - - - -

                                        # Plotting all the points
                                        if X_train.shape[1] == 1 and plot_flag and False:
                                            # with plt.style.context(['science', 'grid']):
                                            n_repeat_plt = 25
                                            fontsize = 18
                                            fig, axe = plt.subplots(figsize=(8.4, 8.4))
                                            plt.rcParams.update({'font.size': fontsize})
                                            y_pred = np.zeros([X_test.shape[0], n_repeat_plt])
                                            sort_idxs_test = np.argsort(X_test[:, 0])
                                            sort_idxs_train = np.argsort(X_train[:, 0])
                                            for _n in range(0, n_repeat_plt):
                                                y_pred[:, _n] = noisy_reg.predict(X_test[sort_idxs_test])
                                            X_test_ravel = np.repeat(X_test[sort_idxs_test, 0], n_repeat_plt)
                                            plt.plot(X_test_ravel, y_pred.ravel(), linestyle='', marker='x', color='r', label='Test: Noisy prediction', markersize=6.0, alpha=0.25)
                                            plt.plot(X_train[sort_idxs_train, 0], y_train[sort_idxs_train, 0], linestyle='-', label='Train', linewidth=8.0)
                                            plt.plot(X_test[sort_idxs_test,0], y_test[sort_idxs_test, 0], linestyle='-', color='k', label='Test: Ground truth', linewidth=2.0)
                                            plt.plot(X_test[sort_idxs_test,0], pred_cln[sort_idxs_test], linestyle='', marker='o', color='k', label='Test: Noiseless prediction', linewidth=2.0, markersize=6.0)
                                            plt.xlabel('x')
                                            plt.ylabel('y')
                                            plt.title('Regression with ensemble of $T='+str(_m)+'$: \n$'+str(round(100*(1-noisless_fraction)))+'$% noisy sub-regressors at $SNR='+str(snr_db)+'$[dB]')
                                            handles, labels = plt.gca().get_legend_handles_labels()
                                            order = [1, 2, 3, 0]
                                            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
                                            plt.grid()
                                            mse_cln, mae_cln = aux.calc_error(y_test[sort_idxs_test, 0], pred_cln[sort_idxs_test], 'mse'), aux.calc_error(y_test[sort_idxs_test, 0], pred_cln[sort_idxs_test], 'mae')
                                            y_test_ravel = np.repeat(y_test[sort_idxs_test, 0], n_repeat_plt)
                                            mse_pred, mae_pred = aux.calc_error(y_test_ravel, y_pred.ravel(), 'mse'), aux.calc_error(y_test_ravel, y_pred.ravel(), 'mae')
                                            if data_type=="sin":
                                                yloc = -2.0
                                            elif data_type=="exp":
                                                yloc = -0.5
                                            plt.text(0, yloc, "MSE (Noiseless): "+"{0:.1f}".format(10*np.log10(mse_cln))+"[dB]" + "\nMSE (Noisy): "+"{0:.1f}".format(10*np.log10(mse_pred))+"[dB]\n"\
                                                            +"MAE (Noiseless): "+"{0:.1f}".format(10*np.log10(mae_cln))+"[dB]" + "\nMAE (Noisy): "+"{0:.1f}".format(10*np.log10(mae_pred))+"[dB]",
                                                     fontsize=fontsize,
                                                     bbox=dict(facecolor='green', alpha=0.1))
                                            plt.show(block=False)

                                        # Predicting with noise
                                        pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                        for _n in range(0, n_repeat):
                                                # - - - NON-ROBUST - - -
                                                pred_nr = noisy_reg.predict(X_test)
                                                err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_nr, criterion)
                                                # err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(pred_cln, pred_nr, criterion)
                                                # - - - ROBUST - - -
                                                pred_r = noisy_rreg.predict(X_test)
                                                err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_r, criterion)
                                                # err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(pred_cln, pred_r, criterion)

                                        # Expectation of error (over multiple realizations)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                        err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                        print("Error [dB], (Clean, Non-robust, Robust) = (" +
                                              "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                                              )

                                        # Sample presentation of data
                                        if X_train.shape[1] == 1 and plot_flag and False:
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

                        if save_results_to_file_flag:
                                results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                                        'Bagging, Noiseless': pd.Series(10*np.log10(err_cln[:, _m_idx, :].mean(1))),
                                                        'Bagging, Non-Robust': pd.Series(10*np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                        'Bagging, Robust': pd.Series(10*np.log10(err_r[:, _m_idx, :].mean(1)))},
                                                       axis=1)
                                results_df.to_csv(results_path + data_type + "_bagging_" + bagging_method + "_" + _m.__str__() + "_" + criterion + "_" + sigma_profile_type + ".csv")

                        if plot_flag:
                                # Plot error results
                                for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, 10 * np.log10(err_cln[:, _m_idx, :].mean(1)), '-k', label='Clean')
                                        plt.plot(snr_db_vec, 10 * np.log10(err_nr[:, _m_idx, :].mean(1)), '-xr',
                                                 label='Non-robust')
                                        plt.plot(snr_db_vec, 10 * np.log10(err_r[:, _m_idx, :].mean(1)), '-ob', label='Robust')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(
                                                _m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]')
                                        plt.ylabel(criterion.upper()+' [dB]')
                                        plt.legend()
                                        plt.show(block=False)
                                # Plot error gain results
                                for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, 10 * np.log10(err_nr[:, _m_idx, :].mean(1)) - 10 * np.log10(
                                                err_r[:, _m_idx, :].mean(1)), '-ob', label='Robust')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(
                                                _m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]')
                                        plt.ylabel(criterion.upper()+' Gain [dB]')
                                        plt.show(block=False)
                        print("---------------------------------------------------------------------------\n")

