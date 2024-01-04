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
results_path = "Results//"

ensemble_size = [16]  # [16, 64] # [5] # Number of weak-learners
tree_max_depth = 5  # Maximal depth of decision tree
min_sample_leaf = 1

data_type_vec = ["sin", "exp", "diabetes", "make_reg", "white-wine", "kc_house_data"]
KFold_n_splits = 4  # Number of k-fold x-validation dataset splits
n_repeat = 75  # Number of iterations for estimating expected performance
n_samples = 1000  # Size of the (synthetic) dataset  in case of synthetic dataset
train_noise = 0.01  # Standard deviation of the measurement / training noise in case of synthetic dataset

gradboost_robust_flag = True

# Dataset specific params for Gradient-descent and other stuff
def getGradDecParams(reg_algo):
    if reg_algo == "Bagging":
        gd_learn_rate_dict = {  # learning rate for grad-dec per dataset: MAE, Bagging, BEM/GEM
            "sin": 1e-4,
            "sin_outliers": 1e-4,
            "exp": 1e-4,
            "make_reg": 1e-4,
            "diabetes": 1e-4,
            "white-wine": 1e-4,
            "kc_house_data": 1e-4
        }
        gd_tol = 1e-2  #
        gd_decay_rate = 0.0  #
        gd_learn_rate_dict_r = gd_learn_rate_dict
        bag_regtol_dict = {
            "sin": 1e-9,
            "sin_outliers": 1e-9,
            "exp": 1e-9,
            "make_reg": 1e-15, # doesnt affect results
            "diabetes": 1e-9,
            "white-wine": 1e-12,
            "kc_house_data": 1e-2
        }
    elif reg_algo == "GradBoost":
        gd_learn_rate_dict = {  # learning rate for grad-dec per dataset: MAE, GradBoost - NonRobust
            "sin": 1e-2,
            "sin_outliers": 1e-2,
            "exp": 1e-2,
            "make_reg": 1e-2,
            "diabetes": 1e-2,
            "white-wine": 1e-2,
            "kc_house_data": 1e-2
        }
        gd_learn_rate_dict_r = {  # learning rate for grad-dec per dataset: MAE, GradBoost - Robust
            "sin": 1e-2,
            "sin_outliers": 1e-2,
            "exp": 1e-2,
            "make_reg": 1e-2,
            "diabetes": 1e-2,
            "white-wine": 1e-2,
            "kc_house_data": 1e-2
        }
        gd_tol = 1e-6  #
        gd_decay_rate = 0.0  #
        bag_regtol_dict = []
    return gd_learn_rate_dict, gd_learn_rate_dict_r, gd_tol, gd_decay_rate, bag_regtol_dict

def get_noise_covmat(y_train, _m=1, snr_db=0, noisy_scale=1):
    snr = 10 ** (snr_db / 10)
    sig_var = np.mean(np.abs(y_train) ** 2)  # np.var(y_train)
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
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

####################################################
####################################################

####################################################
# 1: Distribution of coefficients across Bagging ensembles
####################################################
enable_flag_1 = True
if enable_flag_1:
    reg_algo, bagging_method, criterion = "Bagging", "lr", "mse"
    gd_learn_rate_dict, gd_learn_rate_dict_r, gd_tol, gd_decay_rate, bag_regtol_dict = getGradDecParams(reg_algo)
    ensemble_size, tree_max_depth, min_sample_leaf = [20], 5, 1
    data_type_vec = ["sin", "diabetes"]
    sigma_profile_type_vec = ["uniform", "noiseless_even"]
    _m = ensemble_size[0]
    snr_db, noisy_scale = 10, 20
    plot_flag = True

    for _profile_idx, sigma_profile_type in enumerate(sigma_profile_type_vec):
        coefs_nr, coefs_r = [], []
        for _ds_idx, data_type in enumerate(data_type_vec):
            print("- - - dataset: " + str(data_type) + " - - -")
            # Dataset preparation
            X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
            kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

            y_train_avg, y_test_avg = [], []
            coefs_nr_per_fold, coefs_r_per_fold = np.zeros([_m, kf.n_splits]), np.zeros([_m, kf.n_splits])
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
                # Initiating the tree
                if criterion == "mse":
                    cln_reg = sklearn.ensemble.BaggingRegressor(
                            sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                            n_estimators=_m, random_state=0)
                elif criterion == "mae":
                    cln_reg = sklearn.ensemble.BaggingRegressor(
                            sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth,
                                                          criterion="mae"),
                            n_estimators=_m, random_state=0)

                # Fitting on training data
                cln_reg.fit(X_train, y_train[:, 0])
                # - - - - - - - - - - - - - - - - -

                print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                # Set noise variance
                noise_covariance = get_noise_covmat(y_train, _m, snr_db, noisy_scale)

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

                coefs_r_per_fold[:, kfold_idx] = noisy_reg.weights
                coefs_nr_per_fold[:, kfold_idx] = noisy_rreg.weights

            coefs_nr.append(coefs_nr_per_fold)
            coefs_r.append(coefs_r_per_fold)

            # Plotting distributions of coefficients
            if _profile_idx == 0 and _ds_idx == 0:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
                axes_flat = axes.flatten()
                colors = ['blue', 'red', 'green', 'tan']
                hatches = ["//", "++", "..", "xx"]
                edges = np.arange(-0.150, 0.275, 0.025/2)
            ax = axes[_ds_idx, _profile_idx]
            n, bins, patches = ax.hist(coefs_nr[_ds_idx], bins=edges, label=data_type, color=colors[0:kf.n_splits], density=True, stacked=True, histtype='bar')
            # ax.hist(coefs_nr[_ds_idx], nbins, label=data_type, color=colors[0:kf.n_splits], density=True, stacked=False, histtype='step', fill=False)
            for patch_set, hatch in zip(patches, hatches):
                for patch in patch_set.patches:
                    patch.set_hatch(hatch)
            ax.set_title(data_type+", "+sigma_profile_type)
            # ax.set_xlabel('Values')
            # ax.set_ylabel('Counts')
            plt.show(block=False)
            # plt.title(sigma_profile_type)
            # plt.setp(axes, xlim=[-0.05, 0.25])
    print("---------------------------------------------------------------------------\n")

####################################################
# 2: rGB vs noisy training
####################################################
enable_flag_2 = False
if enable_flag_2:
    reg_algo, bagging_method, criterion = "GradBoost", "gem", "mse"
    gd_learn_rate_dict, gd_learn_rate_dict_r, gd_tol, gd_decay_rate, bag_regtol_dict = getGradDecParams(reg_algo)
    _m, tree_max_depth, min_sample_leaf = 24, 5, 1
    sigma_profile_type = "noiseless_even"  # uniform / single_noisy / noiseless_even (for GradBoost)
    snr_db_vec, noisy_scale = np.linspace(-25, 25, 10), 20
    data_type_vec = ["kc_house_data"]

    for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")
        print("T=" + str(_m) + " regressors")
        # Dataset preparation
        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

        err_cln = np.zeros((len(snr_db_vec), KFold_n_splits))
        err_nr, err_r, err_nt = np.zeros_like(err_cln), np.zeros_like(err_cln), np.zeros_like(err_cln)
        kfold_idx = 0
        for train_index, test_index in kf.split(X):
            print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
            err_cln[:, kfold_idx] = aux.calc_error(y_test[:, 0], pred_cln, criterion)
            # - - - - - - - - - - - - - - - - -

            for idx_snr_db, snr_db in enumerate(snr_db_vec):
                print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")

                # Set prediction noise variance
                noise_covariance = get_noise_covmat(y_train, _m+1, snr_db, noisy_scale)

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
                rgb_noisytrain = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                        min_sample_leaf=min_sample_leaf,
                                        TrainNoiseCov=noise_covariance,
                                        RobustFlag=False,
                                        criterion=criterion,
                                        gd_tol=gd_tol, gd_learn_rate=gd_learn_rate_dict_r[data_type], gd_decay_rate=gd_decay_rate)

                # Fitting on training data with noise: non-robust, robust, noisy training
                rgb_nr.fit(X_train, y_train, m=_m)
                rgb_r.fit(X_train, y_train, m=_m)
                rgb_noisytrain.fit_mse_noisy(X_train, y_train, m=_m, rng=rng)
                # - - - - - - - - - - - - - - - - -

                # Predicting with noise (for reference)
                pred_nr, pred_r, pred_nt = np.zeros(len(y_test)), np.zeros(len(y_test)), np.zeros(len(y_test))
                for _n in range(0, n_repeat):
                    # - - - NON-ROBUST - - -
                    pred_nr = rgb_nr.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                    err_nr[idx_snr_db, kfold_idx] += aux.calc_error(y_test[:,0], pred_nr, criterion)
                    # - - - ROBUST - - -
                    pred_r = rgb_r.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                    err_r[idx_snr_db, kfold_idx] += aux.calc_error(y_test[:,0], pred_r, criterion)
                    # - - - NOISY TRAINING - - -
                    pred_nt = rgb_noisytrain.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                    err_nt[idx_snr_db, kfold_idx] += aux.calc_error(y_test[:,0], pred_nt, criterion)

                # Expectation of error (over multiple realizations)
                err_nr[idx_snr_db, kfold_idx] /= n_repeat
                err_r[idx_snr_db, kfold_idx] /= n_repeat
                err_nt[idx_snr_db, kfold_idx] /= n_repeat

                print("Error [dB], (Clean, Non-robust, Robust, NoisyTrain) = (" +
                      "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, kfold_idx])) + ", " +
                      "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, kfold_idx])) + ", " +
                      "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, kfold_idx])) + ", " +
                      "{0:0.3f}".format(10 * np.log10(err_nt[idx_snr_db, kfold_idx])) + ")"
                      )

            kfold_idx += 1

        # Plot error comparison
        if True:
            plt.figure(figsize=(12, 8))
            # plt.plot(snr_db_vec, 10 * np.log10(err_cln[:, :].mean(1)), '-k', label='Clean')
            plt.plot(snr_db_vec, err_r[:, :].mean(1), '-ob', label='Robust Gradient Boosting (This work)')
            plt.plot(snr_db_vec, err_nr[:, :].mean(1), '-xr', label='Gradient Boosting (non-robust)')
            plt.plot(snr_db_vec, err_nt[:, :].mean(1), '-dy', label='Noisy Training')
            plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
            plt.xlabel('SNR [dB]')
            plt.ylabel('MSE')
            plt.legend()
            plt.grid(visible=True)
            plt.show(block=False)
        print("---------------------------------------------------------------------------\n")

####################################################
# 3: Evaluate MAE with MSE-optimized vs MAE-optimized weights
####################################################
enable_flag_3 = False
if enable_flag_3:
    gd_learn_rate_dict, gd_learn_rate_dict_r, gd_tol, gd_decay_rate, bag_regtol_dict = getGradDecParams("Bagging")
    bagging_method = "gem"
    ensemble_size, tree_max_depth, min_sample_leaf = [8], 6, 1
    sigma_profile_type = "noiseless_even"  # uniform / single_noisy / noiseless_even (for GradBoost)
    snr_db, noisy_scale = 10, 20
    _m_idx = 0
    _m = ensemble_size[_m_idx]

    data_type_vec = ["sin_outliers"]
    KFold_n_splits = 4

    for data_type in data_type_vec:
        print("- - - dataset: " + str(data_type) + " - - -")
        # Dataset preparation
        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)

        err_cln = np.zeros(KFold_n_splits)
        err_r_mse, err_r_mae, err_mis, err_nr_mae = np.zeros_like(err_cln), np.zeros_like(err_cln), np.zeros_like(err_cln), np.zeros_like(err_cln)
        kfold_idx = 0
        for train_index, test_index in kf.split(X):
            print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # - - - CLEAN GRADIENT BOOSTING - - -
            rgb_cln = sklearn.ensemble.BaggingRegressor(
                sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth,
                                              criterion="mae"),
                n_estimators=_m, random_state=0)
            rgb_cln.fit(X_train, y_train[:, 0])  # Fitting on training data
            pred_cln = rgb_cln.predict(X_test)  # Predicting without noise (for reference)
            err_cln[kfold_idx] = aux.calc_error(y_test[:, 0], pred_cln, "mae")  # Calc MAE

            # - - - MSE / MAE BAGGING - - -
            # Set noise variance
            noise_covariance = get_noise_covmat(y_train, _m, snr_db, noisy_scale)
            # Create Bagging regressor instances
            rgb_r_mse = rBaggReg(rgb_cln, noise_covariance, _m, "robust-" + bagging_method, gd_tol,
                                 gd_learn_rate_dict[data_type], gd_decay_rate, bag_regtol_dict[data_type])
            rgb_r_mae = rBaggReg(rgb_cln, noise_covariance, _m, "robust-" + bagging_method, gd_tol,
                                 0.02, gd_decay_rate, bag_regtol_dict[data_type])
            rgb_nr_mae = rBaggReg(rgb_cln, noise_covariance, _m, bagging_method, gd_tol,
                                  0.02, gd_decay_rate, bag_regtol_dict[data_type])
            # Fit on training data
            rgb_r_mse.fit_mse(X_train, y_train[:, 0])
            rgb_r_mae.fit_mae(X_train, y_train[:, 0])
            rgb_nr_mae.fit_mae(X_train, y_train[:, 0])

            # Predict with noise
            pred_r_mse, pred_r_mae, pred_mis, pred_nr_mae = np.zeros(len(y_test)), np.zeros(len(y_test)), np.zeros(len(y_test)), np.zeros(len(y_test))
            for _n in range(0, n_repeat):
                # - - - MSE - - -
                pred_r_mse = rgb_r_mse.predict(X_test, rng=rng)
                err_r_mse[kfold_idx] += aux.calc_error(y_test[:, 0], pred_r_mse, "mse") / n_repeat
                # - - - MAE weights on MAE criterion - - -
                pred_r_mae = rgb_r_mae.predict(X_test, rng=rng)
                err_r_mae[kfold_idx] += aux.calc_error(y_test[:, 0], pred_r_mae, "mae") / n_repeat
                # - - - MSE weights on MAE criterion - - -
                pred_mis = rgb_r_mae.predict(X_test, rng=rng, weights=rgb_r_mse.weights)
                err_mis[kfold_idx] += aux.calc_error(y_test[:, 0], pred_mis, "mae") / n_repeat
                # - - - Non-robust weights on MAE criterion - - -
                pred_nr_mae = rgb_nr_mae.predict(X_test, rng=rng, weights=rgb_nr_mae.weights)
                err_nr_mae[kfold_idx] += aux.calc_error(y_test[:, 0], pred_nr_mae, "mae") / n_repeat

            # # # # # # # # # # # # # # # # # # # # # # # # # #
            if kfold_idx == 1:
                fig = plt.figure(figsize=(12, 8))
                fig.set_label(data_type + "_example")
                fontsize = 18
                plt.rcParams.update({'font.size': fontsize})
                plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                plt.plot(X_test[:, 0], pred_cln, 'o',
                         label="Noiseless, " + "MAE=" + "{:.3f}".format(
                             err_cln[kfold_idx]))
                plt.plot(X_test[:, 0], pred_r_mae, 'o',
                         label="MAE-optimized Robust Bagging (This work), " + "MAE=" + "{:.3f}".format(
                             err_r_mae[kfold_idx]))
                plt.plot(X_test[:, 0], pred_mis, 'd',
                         label="MSE-optimized Robust Bagging (This work), " + "MAE=" + "{:.3f}".format(
                             err_mis[kfold_idx]))
                plt.plot(X_test[:, 0], pred_nr_mae, 'o',
                         label="Bagging (non-robust), " + "MAE=" + "{:.3f}".format(
                             err_nr_mae[kfold_idx]))
                plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.show(block=False)
            # # # # # # # # # # # # # # # # # # # # # # # # # #

            err_gain = err_mis[kfold_idx] / err_r_mae[kfold_idx]
            print("Gain for dataset "+data_type+" at SNR="+"{0:0.3f}".format(snr_db)+"[dB]: "+"{0:0.3f}".format(err_gain))
            kfold_idx += 1

        print("---------------------------------------------------------------------------\n")

####################################################
# 4: Exemplify value of lambda in time-divided noisy setting
####################################################
enable_flag_4 = False
if enable_flag_4:
    reg_algo, bagging_method = "Bagging", "lr"
    gd_learn_rate_dict, gd_learn_rate_dict_r, gd_tol, gd_decay_rate, bag_regtol_dict = getGradDecParams(reg_algo)
    ensemble_size, tree_max_depth, min_sample_leaf = [8], 5, 1
    data_type_vec, sigma_profile_type = ["sin", "exp", "diabetes", "make_reg", "white-wine", "kc_house_data"], "uniform"
    _m, n_repeat = ensemble_size[0], 100
    snr_db_vec, noisy_scale = [-10], 20
    lamda_vec = np.arange(0, 1, 0.01)

    for data_type_idx, data_type in enumerate(data_type_vec):
        print("- - - dataset: " + str(data_type) + " - - -")
        # Dataset preparation
        X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
        kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)
        err = np.zeros((len(snr_db_vec), KFold_n_splits, len(lamda_vec)))
        kfold_idx = 0
        for train_index, test_index in kf.split(X):
            print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            cln_reg = sklearn.ensemble.BaggingRegressor(
                sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth, criterion="mse"),
                n_estimators=_m, random_state=0)  # CLEAN GRADIENT BOOSTING REGRESSOR
            cln_reg.fit(X_train, y_train[:, 0])  # Fitting on training data
            # Predicting without noise (for reference)
            for idx_snr_db, snr_db in enumerate(snr_db_vec):
                print("snr " + " = " + "{0:0.3f}".format(snr_db) + ": ", end=" ")
                noise_covariance = get_noise_covmat(y_train, _m, snr_db, noisy_scale)  # Set noise covariance
                noisy_rreg = []
                for lidx, lamda in enumerate(lamda_vec):
                    noisy_rreg.append(rBaggReg(cln_reg, noise_covariance, _m, "robust-" + bagging_method, gd_tol,
                                           gd_learn_rate_dict[data_type], gd_decay_rate, bag_regtol_dict[data_type]))
                    noisy_rreg[lidx].fit_mse(X_train, y_train[:, 0], lamda=lamda)
                    # Prediction and error calculation
                    for _n in range(0, n_repeat):
                        if np.mod(_n, 4) == 0:
                            noiseless = False
                        else:
                            noiseless = True
                        pred = noisy_rreg[lidx].predict(X_test, rng=rng, noiseless=noiseless)
                        err[idx_snr_db, kfold_idx, lidx] += aux.calc_error(y_test[:, 0], pred, "mse") / n_repeat
                # # # # # # # # # # # # # # # # # # # # # # # # # #
                for lidx, lamda in enumerate(lamda_vec):
                    print("lmd="+"{0:0.2f}".format(lamda)+": Error = " + "{0:0.3f}".format(np.mean(err[idx_snr_db, :, lidx])))
                kfold_idx += 1
        if data_type_idx == 0:
            fig = plt.figure(figsize=(12, 8))
            fig.set_label(data_type + "_example")
            fontsize = 18
        plt.plot(lamda_vec, np.mean(err[idx_snr_db, :, :], axis=0), label=data_type)
        plt.title("_m=" + "{:d}".format(_m) + ", SNR=" + "{:.2f}".format(snr_db))
        plt.xlabel('\lambda'), plt.ylabel('MSE'), plt.legend(), plt.grid(visible=True)
        plt.rcParams.update({'font.size': fontsize}), plt.show(block=False)

        print("---------------------------------------------------------------------------\n")
