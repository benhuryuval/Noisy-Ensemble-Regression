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
import robustRegressors.auxilliaryFunctions as aux
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constants
rng = np.random.default_rng(seed=42)
plot_flag = True #False
save_results_to_file_flag = True

KFold_n_splits = 4  # Number of k-fold x-validation dataset splits

ensemble_size = [16]  # [16, 64] # [5] # Number of weak-learners
tree_max_depth = 5  # Maximal depth of decision tree
min_sample_leaf = 1

n_snr_pts = 8
snr_db_vec = np.linspace(-6, 16, n_snr_pts)  # simulated SNRs [dB]
# snr_db_vec = np.array([0])
n_repeat = 100  # Number of iterations for estimating expected performance
sigma_profile_type = "uniform"  # uniform / single_noisy / noiseless_even (for GradBoost)
noisy_scale = 1000
n_samples = 1000  # Size of the (synthetic) dataset  in case of synthetic dataset
train_noise = 0.01  # Standard deviation of the measurement / training noise in case of synthetic dataset

data_type_vec = ["sin", "exp", "diabetes", "make_reg", "white-wine", "kc_house_data"] # kc_house_data / diabetes / white-wine / sin / exp / make_reg
data_type_vec = ["kc_house_data"]

criterion = "mae"  # "mse" / "mae"
reg_algo = "Bagging"  # "GradBoost" / "Bagging"
bagging_method = "gem"  # "bem" / "gem" / "lr"
gradboost_robust_flag = True

# Prepare results dir
results_path = "Results//" + ensemble_size[0].__str__() + "_" + criterion + "_" + sigma_profile_type + "_" + reg_algo.lower() + "_" + bagging_method + "//"
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Dataset specific params for Gradient-descent and other stuff
if reg_algo == "Bagging":
    gd_learn_rate_dict = {  # learning rate for grad-dec per dataset: MAE, Bagging, BEM/GEM
        "sin":              1e-2,
        "exp":              1e-2,
        "make_reg":         1e-2,
        "diabetes":         1e-2,
        "white-wine":       1e-2,
        "kc_house_data":    1e-2
    }
    gd_tol = 1e-2  #
    gd_decay_rate = 0.0  #

    bag_regtol_dict = {
        "sin": 1e-9,
        "exp": 1e-9,
        "make_reg": 1e-15, # doesnt affect results
        "diabetes": 1e-9,
        "white-wine": 1e-12,
        "kc_house_data": 1e-2
    }

elif reg_algo == "GradBoost":
    gd_learn_rate_dict = {  # learning rate for grad-dec per dataset: MAE, GradBoost - NonRobust
        "sin": 1e-2,
        "exp": 1e-2,
        "make_reg": 1e-2,
        "diabetes": 1e-2,
        "white-wine": 1e-2,
        "kc_house_data": 1e-2
    }
    gd_learn_rate_dict_r = {  # learning rate for grad-dec per dataset: MAE, GradBoost - Robust
        "sin": 1e-2,
        "exp": 1e-2,
        "make_reg": 1e-2,
        "diabetes": 1e-2,
        "white-wine": 1e-2,
        "kc_house_data": 1e-2
    }
    gd_tol = 1e-6  #
    gd_decay_rate = 0.0  #

def get_noise_covmat(y_train, _m=1, snr_db=0, noisy_scale=1):
    snr = 10 ** (snr_db / 10)
    # sig_var = np.mean(np.abs(y_train) ** 2)  # np.var(y_train)
    sig_var = np.mean(y_train)
    if sigma_profile_type == "uniform":
        sigma_sqr = sig_var / snr
        noise_covariance = np.diag(sigma_sqr * np.ones([_m, ]))
    elif sigma_profile_type == "single_noisy":
        sigma_sqr = sig_var / (snr * (1 + (noisy_scale - 1) / _m))
        noise_covariance = np.diag(sigma_sqr / noisy_scale * np.ones([_m, ]))
        noise_covariance[0, 0] = sigma_sqr
    elif sigma_profile_type == "noiseless_even":
        sigma_sqr = sig_var / ((1 + (noisy_scale - 1) / 2) * snr)
        sigma_profile = sigma_sqr * np.ones([_m, ])
        sigma_profile[1::2] *= noisy_scale
        noise_covariance = np.diag(sigma_profile)
    return noise_covariance

# Verify inputs
if reg_algo == "GradBoost" and bagging_method == "lr":
        raise ValueError('Invalid bagging_method.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print(criterion + ": " + reg_algo + ", " + bagging_method + ", " + sigma_profile_type)

# Main simulation loop(s)
####################################################
# Gradient Boosting
####################################################
if reg_algo == "GradBoost":
        for data_type in data_type_vec:
                print("- - - dataset: " + str(data_type) + " - - -")
                # Dataset preparation
                X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
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
                                # # # Initiating the tree
                                rgb_cln = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                        min_sample_leaf=min_sample_leaf,
                                                        TrainNoiseCov=np.zeros([_m + 1, _m + 1]),
                                                        RobustFlag = gradboost_robust_flag,
                                                        criterion=criterion,
                                                        gd_tol=gd_tol, gd_learn_rate=gd_learn_rate_dict[data_type], gd_decay_rate=gd_decay_rate)

                                # Fitting on training data
                                rgb_cln.fit(X_train, y_train, m=_m)

                                # Predicting without noise (for reference)
                                pred_cln = rgb_cln.predict(X_test, PredNoiseCov=np.zeros([_m + 1, _m + 1]), rng=rng)
                                # # Saving the predictions to the training set
                                # err_cln[:, _m_idx, kfold_idx] = np.abs(np.subtract(y_test[:, 0], pred_cln)).mean()
                                err_cln[:, _m_idx, kfold_idx] = aux.calc_error(y_test[:, 0], pred_cln, criterion)
                                # - - - - - - - - - - - - - - - - -

                                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                        print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                                        # Set noise variance
                                        noise_covariance = get_noise_covmat(y_train, ensemble_size[_m_idx]+1, snr_db, noisy_scale)

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
                                                                gd_tol=gd_tol, gd_learn_rate=gd_learn_rate_dict_r[data_type], gd_decay_rate=gd_decay_rate)

                                        # Fitting on training data with noise: non-robust and robust
                                        rgb_r.fit(X_train, y_train, m=_m)
                                        rgb_nr.fit(X_train, y_train, m=_m)
                                        # - - - - - - - - - - - - - - - - -

                                        # Predicting with noise (for reference)
                                        pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                        for _n in range(0, n_repeat):
                                                # - - - NON-ROBUST - - -
                                                pred_nr = rgb_nr.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                                                err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test[:,0], pred_nr, criterion)
                                                # - - - ROBUST - - -
                                                pred_r = rgb_r.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                                                err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test[:,0], pred_r, criterion)

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
                                                # plt.plot(X_test[:, 0], pred_cln, 'o',
                                                #          label="Clean, " + "err=" + "{:.4f}".format(
                                                #                  err_cln[0, _m_idx, kfold_idx]))
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
                                                        'GradBoost, Noiseless': pd.Series(10 * np.log10(err_cln[:, _m_idx, :].mean(1))),
                                                        'GradBoost, Non-Robust': pd.Series(10 * np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                        'GradBoost, Robust': pd.Series(10 * np.log10(err_r[:, _m_idx, :].mean(1)))},
                                                       axis=1)
                                results_df.to_csv(results_path + _m.__str__() + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_gbr" + ".csv")
                        print("---------------------------------------------------------------------------\n")

                # Plot error and error gain
                if plot_flag:
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
                X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
                kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)
                y_train_avg, y_test_avg = [], []

                for _m_idx, _m in enumerate(ensemble_size):
                        print("T=" + str(_m) + " regressors")

                        err_cln = np.zeros((len(snr_db_vec), len(ensemble_size), KFold_n_splits))
                        err_nr, err_r, err_rcln = np.zeros_like(err_cln), np.zeros_like(err_cln), np.zeros_like(err_cln)

                        lb_1, lb_2 = np.ones((len(snr_db_vec), KFold_n_splits)), np.ones((len(snr_db_vec), KFold_n_splits))
                        ub_bem, ub_gem = np.ones((len(snr_db_vec), KFold_n_splits)), np.ones((len(snr_db_vec), KFold_n_splits))

                        kfold_idx = -1
                        for train_index, test_index in kf.split(X):
                                # - - - Load data
                                print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                kfold_idx += 1

                                y_train_avg.append(np.nanmean(np.abs(y_train)))
                                y_test_avg.append(np.nanmean(np.abs(y_test)))

                                # - - - CLEAN BAGGING - - -
                                # Initiating the ensemble using sklearn
                                if criterion == "mse":
                                    cln_reg = sklearn.ensemble.BaggingRegressor(
                                            sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                                            n_estimators=_m, random_state=0)
                                elif criterion == "mae":
                                    cln_reg = sklearn.ensemble.BaggingRegressor(
                                            sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth,
                                                                          criterion="mae"),
                                            n_estimators=_m, random_state=0)
                                # Fit, predict and calculate error of "clean" ensemble on training data
                                cln_reg.fit(X_train, y_train[:, 0])
                                pred_cln = cln_reg.predict(X_test)
                                err_cln[:, _m_idx, kfold_idx] = aux.calc_error(y_test, pred_cln, criterion)

                                # Initiating the ensemble using rBaggReg
                                ncov = get_noise_covmat(0, ensemble_size[_m_idx], 0, 1)
                                cln_rreg = rBaggReg(cln_reg, ncov, _m, bagging_method, gd_tol, gd_learn_rate_dict[data_type], gd_decay_rate, bag_regtol_dict[data_type])
                                # Fit, predict and calculate error of "clean" ensemble on training data
                                if criterion == "mse":
                                    cln_rreg.fit_mse(X_train, y_train[:, 0])
                                elif criterion == "mae":
                                    cln_rreg.fit_mae(X_train, y_train[:, 0])
                                pred_rcln = cln_rreg.predict(X_test)
                                err_rcln[:, _m_idx, kfold_idx] = aux.calc_error(y_test, pred_rcln, criterion)
                                # - - - - - - - - - - - - - - - - -

                                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                        print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                                        # Set noise variance
                                        # noise_covariance = get_noise_covmat(y_train, ensemble_size[_m_idx], snr_db, noisy_scale)
                                        noise_covariance = get_noise_covmat(err_rcln[:, _m_idx, kfold_idx], ensemble_size[_m_idx], snr_db, noisy_scale)

                                        # - - - NON-ROBUST / ROBUST BAGGING - - -
                                        noisy_reg = rBaggReg(cln_reg,   noise_covariance, _m, bagging_method,           gd_tol, gd_learn_rate_dict[data_type], gd_decay_rate, bag_regtol_dict[data_type])
                                        noisy_rreg = rBaggReg(cln_reg,  noise_covariance, _m, "robust-"+bagging_method, gd_tol, gd_learn_rate_dict[data_type], gd_decay_rate, bag_regtol_dict[data_type])

                                        # Fitting on training data with noise: non-robust and robust
                                        if criterion == "mse":
                                            noisy_reg.fit_mse(X_train, y_train[:, 0])
                                            noisy_rreg.fit_mse(X_train, y_train[:, 0])
                                        elif criterion == "mae":
                                            noisy_reg.fit_mae(X_train, y_train[:, 0])
                                            noisy_rreg.fit_mae(X_train, y_train[:, 0])
                                        # - - - - - - - - - - - - - - - - -

                                        # - - - Calculate lower/upper bounds - - -
                                        if criterion == "mae":
                                            lb_1[idx_snr_db, kfold_idx], lb_2[idx_snr_db, kfold_idx] = noisy_reg.calc_mae_lb(X_train, y_train, weights=noisy_reg.weights)
                                            ub_bem[idx_snr_db, kfold_idx], ub_gem[idx_snr_db, kfold_idx] = noisy_reg.calc_mae_ub(X_train, y_train)
                                            # ub[idx_snr_db, kfold_idx] = np.nanmin([ub_bem, ub_gem])

                                        # Predicting with noise
                                        pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                        for _n in range(0, n_repeat):
                                                # - - - NON-ROBUST - - -
                                                pred_nr = noisy_reg.predict(X_test, rng=rng)
                                                err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_nr, criterion)
                                                # - - - ROBUST - - -
                                                pred_r = noisy_rreg.predict(X_test, rng=rng)
                                                err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_r, criterion)

                                        # Expectation of error (over multiple realizations)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                        err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                        print("Error [dB], (Clean, Clean (r), Non-robust, Robust) = (" +
                                              "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_rcln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                                              )
                                        if criterion == "mae":
                                            print("Bounds (MAE) [dB], (Lower, Upper(BEM), Upper(GEM)) = (" +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([lb_1[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([lb_2[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([ub_bem[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([ub_gem[idx_snr_db, kfold_idx], 1e-10]))) + ")"
                                              )

                                        # Sample presentation of data
                                        if X_train.shape[1] == 1 and plot_flag and False:
                                                fig = plt.figure(figsize=(12, 8))
                                                fig.set_label(data_type + "_example")
                                                fontsize = 18
                                                plt.rcParams.update({'font.size': fontsize})
                                                plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                                                plt.plot(X_test[:, 0], pred_cln, 'o',
                                                         label="Clean, " + "err=" + "{:.4f}".format(
                                                                 err_cln[0, _m_idx, kfold_idx]))
                                                plt.plot(X_test[:, 0], pred_rcln, 'p',
                                                         label="Clean (r), " + "err=" + "{:.4f}".format(
                                                                 err_rcln[0, _m_idx, kfold_idx]))
                                                plt.plot(X_test[:, 0], pred_r, '*',
                                                         label="Robust, " + "err=" + "{:.4f}".format(
                                                                 err_r[idx_snr_db, _m_idx, kfold_idx]))
                                                plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
                                                plt.plot(X_test[:, 0], pred_nr, 'd',
                                                         label="Non-Robust, " + "err=" + "{:.4f}".format(
                                                                 err_nr[idx_snr_db, _m_idx, kfold_idx]))
                                                plt.xlabel('x'), plt.ylabel('y'), plt.legend()
                                                plt.show(block=False)
                                                plt.close(fig)

                        if save_results_to_file_flag:
                                results_df = pd.concat({'SNR':                          pd.Series(snr_db_vec),
                                                        'Bagging, Noiseless':           pd.Series(10 * np.log10(err_rcln[:, _m_idx, :].mean(1))),
                                                        'Bagging, Non-Robust':          pd.Series(10 * np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                        'Bagging, Robust':              pd.Series(10 * np.log10(err_r[:, _m_idx, :].mean(1))),
                                                        'Lower bound (CLN), Robust':    pd.Series(10 * np.log10(lb_1.mean(1))),
                                                        'Lower bound (FLD), Robust':    pd.Series(10 * np.log10(lb_2.mean(1))),
                                                        'Upper bound (BEM), Robust':    pd.Series(10 * np.log10(ub_bem.mean(1))),
                                                        'Upper bound (GEM), Robust':    pd.Series(10 * np.log10(ub_gem.mean(1))),
                                                        'y Train Avg':                  pd.Series(10 * np.log10([np.mean(y_train_avg)] * n_snr_pts)),
                                                        'y Test Avg':                   pd.Series(10 * np.log10([np.mean(y_test_avg)] * n_snr_pts))
                                                        }, axis=1)
                                results_df.to_csv(results_path + _m.__str__() + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_bagging_" + bagging_method + ".csv")

                        if plot_flag:
                                # Plot error results
                                for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, err_cln[:, _m_idx, :].mean(1), '--k', label='Clean')
                                        plt.plot(snr_db_vec, err_rcln[:, _m_idx, :].mean(1), '-k', label='Clean (r)')
                                        plt.plot(snr_db_vec, err_nr[:, _m_idx, :].mean(1), '-xr', label='Non-robust')
                                        plt.plot(snr_db_vec, err_r[:, _m_idx, :].mean(1), '-ob', label='Robust')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]'), plt.ylabel(criterion.upper())
                                        plt.legend()
                                        plt.show(block=False)
                                # Plot error gain results
                                for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, (err_nr[:, _m_idx, :].mean(1) - err_r[:, _m_idx, :].mean(1)) / err_rcln[-1, _m_idx, :].mean(0), '-ob', label='Robust')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(
                                                _m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]'), plt.ylabel(criterion.upper()+' Gain')
                                        plt.show(block=False)
                                # Plot error bounds for mae
                                if criterion == "mae":
                                    for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, ub_bem.mean(1), '--g', marker='o', label='Upper bound')
                                        plt.plot(snr_db_vec, ub_gem.mean(1), '--g', marker='x', label='Upper bound')
                                        plt.plot(snr_db_vec, lb_1.mean(1), '-g', marker='o', label='Lower bound')
                                        plt.plot(snr_db_vec, lb_2.mean(1), '-g', marker='x', label='Lower bound')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]'), plt.ylabel(criterion.upper())
                                        plt.legend()
                                        plt.show(block=False)
                        print("---------------------------------------------------------------------------\n")

