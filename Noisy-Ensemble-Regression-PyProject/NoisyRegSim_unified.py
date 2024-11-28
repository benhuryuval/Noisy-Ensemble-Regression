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
def mag2db(x, crit="mse"):
    if crit.upper() == "MSE":
        return 10*np.log10(x)
    else:
        return 20 * np.log10(x)
def db2mag(x, crit="mse"):
    if crit.upper() == "MSE":
        return 10**(x/10)
    else:
        return 10**(x/20)

# Constants
rng = np.random.default_rng(seed=42)
plot_flag = False  #True #False
save_results_to_file_flag = True

KFold_n_splits = 4  # Number of k-fold x-validation dataset splits
n_snr_pts = 10

snr_db_vec = np.linspace(-3, 10, n_snr_pts)  # simulated SNRs [dB]
# snr_db_vec = np.array([18])

criterion = "mse"  # "mse" / "mae"
reg_algo = "Bagging"  # "GradBoost" / "Bagging"
bagging_method = "lr"  # "bem" / "gem" / "lr"

n_repeat = 100  # Number of iterations for estimating expected performance
sigma_profile_type = "uniform"  # uniform / single_noisy / noiseless_even
noisy_scale = 20
n_samples = 1000  # Size of the (synthetic) dataset  in case of synthetic dataset
train_noise = 0.01  # Standard deviation of the measurement / training noise in case of synthetic dataset

data_type_vec = ["sin", "make_reg", "diabetes", "white-wine", "kc_house_data"]  # kc_house_data / diabetes / white-wine / sin / exp / make_reg
# data_type_vec = ["sin", "diabetes"]

gradboost_robust_flag = True

# Prepare results dir
results_path = "Results//" + criterion + "_" + sigma_profile_type + "_" + reg_algo.lower() + "_" + bagging_method + "//"
if not os.path.exists(results_path):
    os.mkdir(results_path)

# Dataset specific params for Gradient-descent and other stuff
if reg_algo == "Bagging":
    if criterion == "mse":
        params = {
            "sin":          {
                                "ensemble_size": [32],
                                "tree_max_depth": 4,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
            "exp":          {
                                "ensemble_size": [32],
                                "tree_max_depth": 3,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
            "make_reg":     {
                                "ensemble_size": [32],
                                "tree_max_depth": 2,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
            "diabetes":     {
                                "ensemble_size": [32],
                                "tree_max_depth": 1,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
            "white-wine":   {
                                "ensemble_size": [32],
                                "tree_max_depth": 2,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
            "kc_house_data":{
                                "ensemble_size": [32],
                                "tree_max_depth": 2,
                                "learn_rate": 1e-4,
                                "bag_regtol": 1e-4
                            },
        }
    elif criterion == "mae":
        params = {
            "sin":          {
                                "ensemble_size": [8],
                                "tree_max_depth": 4,
                                "learn_rate": 1e-2,
                                "bag_regtol": 1e-4
                            },
            "exp":          {
                                "ensemble_size": [1],
                                "tree_max_depth": 1,
                                "learn_rate": 5e-2,
                                "bag_regtol": 1e-4
                            },
            "make_reg":     {
                                "ensemble_size": [8],
                                "tree_max_depth": 2,
                                "learn_rate": 5e-2,
                                "bag_regtol": 1e-4
                            },
            "diabetes":     {
                                "ensemble_size": [8],
                                "tree_max_depth": 1,
                                "learn_rate": 0.5,
                                "bag_regtol": 1e-4
                            },
            "white-wine":   {
                                "ensemble_size": [8],
                                "tree_max_depth": 1,
                                "learn_rate": 1,
                                "bag_regtol": 1e-4
                            },
            "kc_house_data":{
                                "ensemble_size": [8],
                                "tree_max_depth": 2,
                                "learn_rate": 5e-2,
                                "bag_regtol": 1e-4
                            },
        }

    gd_tol = 1e-2  #
    gd_decay_rate = 0.0  #

elif reg_algo == "GradBoost":

    params = {
        "sin": {
            "ensemble_size": [1,3,5,7,11,15,25,31],  # [1,2,4,8,12,16,24,32],
            "tree_max_depth": 1,
            "learn_rate": 2e-2,
            "gd_tol": 1e-6
        },
        "exp": {
            "ensemble_size": [8],
            "tree_max_depth": 1,
            "learn_rate": 1e-2,
            "gd_tol": 1e-6
        },
        "make_reg": {
            "ensemble_size": [16],
            "tree_max_depth": 1,
            "learn_rate": 1e-2,
            "gd_tol": 1e-6
        },
        "diabetes": {
            "ensemble_size": [1,2,4,8,12,16,24,32],
            "tree_max_depth": 1,
            "learn_rate": 1e-2,
            "gd_tol": 1e-6
        },
        "white-wine": {
            "ensemble_size": [16],
            "tree_max_depth": 1,
            "learn_rate": 1e-2,
            "gd_tol": 1e-6
        },
        "kc_house_data": {
            "ensemble_size": [32],
            "tree_max_depth": 1,
            "learn_rate": 1e-2,
            "gd_tol": 1e-6
        },
    }

    gd_decay_rate = 0.0  #
    min_sample_leaf = 1

def get_noise_covmat(sig_var, _m=1, snr_db=0, noisy_scale=1):
    snr = 10 ** (snr_db / 10)
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
        sigma_profile[1::2] *= noisy_scale  # INDEX 1 3 5 7 ...
        # sigma_profile[0::2] *= noisy_scale  # INDEX 0 2 4 6...
        noise_covariance = np.diag(sigma_profile)
    return noise_covariance

# Verify inputs
if reg_algo == "GradBoost" and bagging_method == "lr":
    raise ValueError('Invalid bagging_method.')
if reg_algo == "Bagging" and bagging_method == "gem" and criterion == "mse":
    raise ValueError('Unstable bagging_method. Change to lr')

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

                ensemble_size = params[data_type]["ensemble_size"]
                tree_max_depth = params[data_type]["tree_max_depth"]
                learn_rate = params[data_type]["learn_rate"]
                gd_tol = params[data_type]["gd_tol"]

                err_cln = np.zeros((len(snr_db_vec), len(ensemble_size), KFold_n_splits))
                err_nr, err_r = np.zeros_like(err_cln), np.zeros_like(err_cln)
                err_r_cln = np.zeros_like(err_cln)

                for _m_idx, _m in enumerate(ensemble_size):  # iterate number of trees
                        print("T=" + str(_m) + " regressors")

                        kfold_idx = 0
                        for train_index, test_index in kf.split(X):
                                print("\nTRAIN:", train_index[0], " to ", train_index[-1], "\nTEST:", test_index[0], " to ", test_index[-1])
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]

                                # Plotting all the points
                                # if X_train.shape[1] == 1 and plot_flag:
                                #     plt.figure(figsize=(12, 8))
                                #     plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                                #     plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

                                # - - - CLEAN GRADIENT BOOSTING - - -
                                # # # Initiating the tree
                                rgb_cln = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                        min_sample_leaf=min_sample_leaf,
                                                        TrainNoiseCov=np.zeros([_m + 1, _m + 1]),
                                                        RobustFlag = gradboost_robust_flag,
                                                        criterion=criterion,
                                                        gd_tol=gd_tol, gd_learn_rate=learn_rate, gd_decay_rate=gd_decay_rate)

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
                                        noise_covariance = get_noise_covmat(np.mean(np.abs(y_train)**2), ensemble_size[_m_idx]+1, snr_db, noisy_scale)
                                        # noise_covariance = get_noise_covmat(np.mean(err_cln[:, _m_idx, kfold_idx]), ensemble_size[_m_idx]+1, snr_db, noisy_scale)

                                        # - - - NON-ROBUST / ROBUST GRADIENT BOOSTING - - -
                                        rgb_nr = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                min_sample_leaf=min_sample_leaf,
                                                                TrainNoiseCov=np.zeros([_m + 1, _m + 1]),
                                                                RobustFlag=gradboost_robust_flag,
                                                                criterion=criterion,
                                                                gd_tol=gd_tol, gd_learn_rate=learn_rate, gd_decay_rate=gd_decay_rate)
                                        rgb_r = rGradBoost(X=X_train, y=y_train, max_depth=tree_max_depth,
                                                                min_sample_leaf=min_sample_leaf,
                                                                TrainNoiseCov=noise_covariance,
                                                                RobustFlag=gradboost_robust_flag,
                                                                criterion=criterion,
                                                                gd_tol=gd_tol, gd_learn_rate=learn_rate, gd_decay_rate=gd_decay_rate)

                                        # Fitting on training data with noise: non-robust and robust
                                        rgb_r.fit(X_train, y_train, m=_m)
                                        rgb_nr.fit(X_train, y_train, m=_m)
                                        # rgb_nr.gamma[1::2]=0
                                        # - - - - - - - - - - - - - - - - -

                                        # Predicting with noise (for reference)
                                        pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                        pred_r_cln = np.zeros(len(y_test))
                                        for _n in range(0, n_repeat):
                                                # - - - NON-ROBUST - - -
                                                pred_nr = rgb_nr.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                                                err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test[:,0], pred_nr, criterion)
                                                # - - - ROBUST - - -
                                                pred_r = rgb_r.predict(X_test, PredNoiseCov=noise_covariance, rng=rng)
                                                err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test[:,0], pred_r, criterion)

                                                pred_r_cln = rgb_r.predict(X_test, PredNoiseCov=np.zeros([_m + 1, _m + 1]), rng=rng)
                                                err_r_cln[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test[:, 0], pred_r_cln, criterion)

                                        # Expectation of error (over multiple realizations)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                        err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                        err_r_cln[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                        # Sample presentation of data
                                        if X_train.shape[1] == 1 and plot_flag:
                                                fig_dataset = plt.figure(figsize=(12, 8))
                                                # plt.plot(X_train[:, 0], y_train, 'x', label="Train")
                                                plt.plot(X_test[:, 0], y_test, 'x', label="Train")
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

                                        print("\t\t\t\tError, (Clean, Non-robust, Robust) = (" +
                                              "{0:0.3f}".format(err_cln[idx_snr_db, _m_idx, kfold_idx]) + ", " +
                                              "{0:0.3f}".format(err_nr[idx_snr_db, _m_idx, kfold_idx]) + ", " +
                                              "{0:0.3f}".format(err_r[idx_snr_db, _m_idx, kfold_idx]) + ")"
                                              )
                                        print("\t\t\t\tError reduction [%], (NonRobust-Robust)/Clean = " +
                                              "{0:.3f}".format( 100 * (err_nr[idx_snr_db, _m_idx, kfold_idx] - err_r[idx_snr_db, _m_idx, kfold_idx]) /
                                                                 err_cln[idx_snr_db, _m_idx, kfold_idx] )
                                              )
                                        print("\t\t\t\tError degradation [%], (Robust-Clean)/Clean = " +
                                              "{0:.3f}".format( 100 * (err_r[idx_snr_db, _m_idx, kfold_idx] - err_cln[idx_snr_db, _m_idx, kfold_idx]) /
                                                                 err_cln[idx_snr_db, _m_idx, kfold_idx] )
                                              )

                                kfold_idx += 1

                        print("\nOVERALL PERFORMANCE: "+reg_algo+", "+criterion.upper()+", "+data_type+", "+sigma_profile_type+", T="+str(_m))
                        err_cln_ser, err_nr_ser, err_r_ser = pd.Series(err_cln[:, _m_idx, :].mean(1)), pd.Series(err_nr[:, _m_idx, :].mean(1)), pd.Series(err_r[:, _m_idx, :].mean(1))
                        err_r_cln_ser = pd.Series(err_r_cln[:, _m_idx, :].mean(1))
                        results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                                'Noiseless error': err_cln_ser,
                                                'Non-Robust error': err_nr_ser,
                                                'Robust error': err_r_ser,
                                                'Robust error (Clean)': err_r_cln_ser,
                                                'Reduction [%]': 100 * (err_nr_ser - err_r_ser) / err_cln_ser,
                                                'Degradation [%]': 100 * (err_r_ser - err_cln_ser) / err_cln_ser
                                                },
                                            axis=1)
                        pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 200)
                        print(results_df)
                        # print("\nROBUST WEIGHTS:")
                        # print("\nNon-robust:"), print(rgb_nr.gamma)
                        # print("\nRobust:"),\
                        # print(rgb_r.gamma)

                        if save_results_to_file_flag:
                                results_df = pd.concat({'SNR': pd.Series(snr_db_vec),
                                                        'GradBoost, Noiseless': pd.Series(10 * np.log10(err_cln[:, _m_idx, :].mean(1))),
                                                        'GradBoost, Non-Robust': pd.Series(10 * np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                        'GradBoost, Robust': pd.Series(10 * np.log10(err_r[:, _m_idx, :].mean(1))),
                                                        'GradBoost, Robust (No noise)': pd.Series(10 * np.log10(err_r_cln[:, _m_idx, :].mean(1)))
                                                        },
                                                       axis=1)
                                results_df.to_csv(results_path + criterion + "_" + sigma_profile_type + "_" + data_type + "_gbr" + ".csv")
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

                    # Plot error results
                    for _m_idx, _m in enumerate(ensemble_size):
                        plt.figure(figsize=(12, 8))
                        plt.plot(snr_db_vec, err_cln[:, _m_idx, :].mean(1), '--k', label='Clean')
                        plt.plot(snr_db_vec, err_nr[:, _m_idx, :].mean(1), '-xr', label='Non-robust')
                        plt.plot(snr_db_vec, err_r[:, _m_idx, :].mean(1), '-ob', label='Robust')
                        plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                        plt.xlabel('SNR [dB]'), plt.ylabel(criterion.upper())
                        plt.legend()
                        plt.show(block=False)
                    # Plot error gain results
                    for _m_idx, _m in enumerate(ensemble_size):
                        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
                        ax1.set_xlabel('SNR [dB]', fontsize=18)
                        ax1.set_ylabel(criterion.upper() + ' Reduction [%]', fontsize=18)
                        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                        ax2.set_ylabel(criterion.upper() + ' Degradation [%]',
                                       fontsize=18)  # we already handled the x-label with ax1

                        gain = 100 * (err_nr[:, _m_idx, :].mean(1) - err_r[:, _m_idx, :].mean(1)) / err_cln[-1, _m_idx, :].mean(0)
                        deg = 100 * (err_r[:, _m_idx, :].mean(1) - err_cln[-1, _m_idx, :].mean(0)) / err_cln[-1, _m_idx, :].mean(0)
                        ax1.plot(snr_db_vec, gain, '-ob')
                        ax1.set_title("dataset: " + str(data_type) + ", T=" + str(
                            _m) + " regressors\nnoise=" + sigma_profile_type + ", Model error=" + "{:.3f}".format(err_cln[-1, _m_idx, :].mean(0)))
                        ax2.plot(snr_db_vec, deg, '-xb')
                        plt.show(block=False)
                        for ii, xx in enumerate(snr_db_vec):
                            ax1.annotate(
                                "{:.2f}".format(err_nr[ii, _m_idx, :].mean()),
                                xy=(xx, gain[ii]), xycoords='data',
                                xytext=(0, 0), textcoords='offset points',
                                fontsize=10)
                            ax2.annotate(
                                "{:.2f}".format(err_r[ii, _m_idx, :].mean()),
                                xy=(xx, deg[ii]), xycoords='data',
                                xytext=(0, 0), textcoords='offset points',
                                fontsize=10)

                        ax1.set_ylim(bottom=0)
                        # ax2.tick_params(axis='y', labelcolor=color)
                        # ax2.set_ylim(ax1.get_ylim())
                        # We change the fontsize of minor ticks label
                        ax1.tick_params(axis='both', which='major', labelsize=16)
                        ax1.tick_params(axis='both', which='minor', labelsize=16)
                        ax2.tick_params(axis='both', which='major', labelsize=16)
                        ax2.tick_params(axis='both', which='minor', labelsize=16)
                        fig.tight_layout()  # otherwise the right y-label is slightly clipped
                    # Error versus T for given SNR

                data_label = {
                    "sin": "Sine",
                    "exp": "Exp",
                    "make_reg": "Hyperplane",
                    "kc_house_data": "King County",
                    "diabetes": "Diabetes",
                    "white-wine": "Wine"
                }
                fontsize = 16
                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
                    plt.plot(ensemble_size, err_cln[idx_snr_db, :, :].mean(1), '--k', label='Clean')
                    plt.plot(ensemble_size, err_nr[idx_snr_db, :, :].mean(1), '-xr', label='Non-robust')
                    plt.plot(ensemble_size, err_r[idx_snr_db, :, :].mean(1), '-ob', label='Robust')
                    plt.plot(ensemble_size, err_r_cln[idx_snr_db, :, :].mean(1), '-dg', label='Robust (Clean)')
                    plt.title("dataset: " + str(data_type) + ", SNR=" + str(snr_db) + "dB\nnoise=" + sigma_profile_type)
                    ax.set_xlabel('T', fontsize=fontsize), ax.set_ylabel(criterion.upper(), fontsize=fontsize)
                    plt.legend(), ax.grid()
                    plt.show(block=False)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize)
                    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
                    fig.tight_layout()  # otherwise the right y-label is slightly clipped

####################################################
# Bagging
####################################################
noisetype = 'gaussian'  # gaussian / laplace / tstudent
if reg_algo == "Bagging":
        for data_type in data_type_vec:
                print("- - - dataset: " + str(data_type) + " - - -")
                # Dataset preparation
                X, y = aux.get_dataset(data_type=data_type, n_samples=n_samples, noise=train_noise, rng=rng)
                kf = KFold(n_splits=KFold_n_splits, random_state=None, shuffle=False)
                y_train_avg, y_test_avg = [], []

                ensemble_size = params[data_type]["ensemble_size"]
                tree_max_depth = params[data_type]["tree_max_depth"]
                learn_rate = params[data_type]["learn_rate"]
                bag_regtol = params[data_type]["bag_regtol"]

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

                                # Fit, predict and calculate error of "clean" ensemble on training data
                                ncov = get_noise_covmat(0, ensemble_size[_m_idx], 0, 1)
                                # Initiating the ensemble using rBaggReg
                                cln_rreg = rBaggReg(cln_reg, ncov, _m, "robust-" + bagging_method, gd_tol,
                                                    learn_rate, gd_decay_rate, bag_regtol)
                                if criterion == "mse":
                                    cln_rreg.fit_mse(X_train, y_train[:, 0])
                                elif criterion == "mae":
                                    cln_rreg.fit_mae(X_train, y_train[:, 0])
                                pred_rcln = cln_rreg.predict(X_test)
                                err_rcln[:, _m_idx, kfold_idx] = aux.calc_error(y_test, pred_rcln, criterion)
                                # - - - - - - - - - - - - - - - - -

                                for idx_snr_db, snr_db in enumerate(snr_db_vec):
                                        print("snr " + " = " + "{0:2.2f}".format(snr_db) + ": ", end=" ")

                                        # Set noise variance
                                        sig_snr = np.mean(np.abs(y_train)**2)
                                        # sig_snr = np.min((err_rcln[-1, _m_idx, kfold_idx], err_cln[-1, _m_idx, kfold_idx]))
                                        noise_covariance = get_noise_covmat(sig_snr, ensemble_size[_m_idx], snr_db, noisy_scale)

                                        # - - - NON-ROBUST / ROBUST BAGGING - - -
                                        noisy_reg = rBaggReg(cln_reg,   noise_covariance, _m, bagging_method,           gd_tol, learn_rate, gd_decay_rate, bag_regtol)
                                        noisy_rreg = rBaggReg(cln_reg,  noise_covariance, _m, "robust-"+bagging_method, gd_tol, learn_rate, gd_decay_rate, bag_regtol)

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
                                            lb_1[idx_snr_db, kfold_idx], lb_2[idx_snr_db, kfold_idx] = noisy_reg.calc_mae_lb(X_train, y_train, weights=cln_rreg.weights)
                                            ub_bem[idx_snr_db, kfold_idx], ub_gem[idx_snr_db, kfold_idx] = noisy_reg.calc_mae_ub(X_train, y_train)
                                            # ub[idx_snr_db, kfold_idx] = np.nanmin([ub_bem, ub_gem])

                                        # Predicting with noise
                                        pred_nr, pred_r = np.zeros(len(y_test)), np.zeros(len(y_test))
                                        for _n in range(0, n_repeat):
                                            # - - - NON-ROBUST - - -
                                            pred_nr = noisy_reg.predict(X_test, rng=rng, noisetype=noisetype)
                                            err_nr[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_nr, criterion)
                                            # - - - ROBUST - - -
                                            pred_r = noisy_rreg.predict(X_test, rng=rng, noisetype=noisetype)
                                            err_r[idx_snr_db, _m_idx, kfold_idx] += aux.calc_error(y_test, pred_r, criterion)

                                        # Expectation of error (over multiple realizations)
                                        err_nr[idx_snr_db, _m_idx, kfold_idx] /= n_repeat
                                        err_r[idx_snr_db, _m_idx, kfold_idx] /= n_repeat

                                        print("\tError [dB], (Clean, Clean (r), Non-robust, Robust) = (" +
                                              "{0:0.3f}".format(10 * np.log10(err_cln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_rcln[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_nr[idx_snr_db, _m_idx, kfold_idx])) + ", " +
                                              "{0:0.3f}".format(10 * np.log10(err_r[idx_snr_db, _m_idx, kfold_idx])) + ")"
                                              )
                                        print("\t\t\t\tError, (Clean, Clean (r), Non-robust, Robust) = (" +
                                              "{0:0.3f}".format(err_cln[idx_snr_db, _m_idx, kfold_idx]) + ", " +
                                              "{0:0.3f}".format(err_rcln[idx_snr_db, _m_idx, kfold_idx]) + ", " +
                                              "{0:0.3f}".format(err_nr[idx_snr_db, _m_idx, kfold_idx]) + ", " +
                                              "{0:0.3f}".format(err_r[idx_snr_db, _m_idx, kfold_idx]) + ")"
                                              )
                                        ref_cln_err = np.min([err_cln[idx_snr_db, _m_idx, kfold_idx], err_rcln[idx_snr_db, _m_idx, kfold_idx]])
                                        print("\t\t\t\tError reduction [%], (NonRobust-Robust)/Clean = " +
                                              "{0:.3f}".format( 100 * (err_nr[idx_snr_db, _m_idx, kfold_idx] - err_r[idx_snr_db, _m_idx, kfold_idx]) /
                                                                 ref_cln_err )
                                              )
                                        print("\t\t\t\tError degradation [%], (Robust-Clean)/Clean = " +
                                              "{0:.3f}".format( 100 * (err_r[idx_snr_db, _m_idx, kfold_idx] - ref_cln_err) /
                                                                 ref_cln_err )
                                              )
                                        if criterion == "mae":
                                            print("\t\t\tBounds (MAE) [dB], (Lower, Upper(BEM), Upper(GEM)) = (" +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([lb_1[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([lb_2[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([ub_bem[idx_snr_db, kfold_idx], 1e-10]))) + ", " +
                                                  "{0:0.3f}".format(10 * np.log10(np.max([ub_gem[idx_snr_db, kfold_idx], 1e-10]))) + ")"
                                              )

                                        # Sample presentation of data
                                        if X_train.shape[1] == 1 and plot_flag and True:
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
                                ref_err = np.min([err_rcln[:, _m_idx, :].mean(1), err_cln[:, _m_idx, :].mean(1)], axis=0)
                                results_df = pd.concat({'SNR':                          pd.Series(snr_db_vec),
                                                        'Bagging, Noiseless':           pd.Series(10 * np.log10(ref_err)),
                                                        'Bagging, Non-Robust':          pd.Series(10 * np.log10(err_nr[:, _m_idx, :].mean(1))),
                                                        'Bagging, Robust':              pd.Series(10 * np.log10(err_r[:, _m_idx, :].mean(1))),
                                                        'Lower bound (CLN), Robust':    pd.Series(10 * np.log10(lb_1.mean(1))),
                                                        'Lower bound (FLD), Robust':    pd.Series(10 * np.log10(lb_2.mean(1))),
                                                        'Upper bound (BEM), Robust':    pd.Series(10 * np.log10(ub_bem.mean(1))),
                                                        'Upper bound (GEM), Robust':    pd.Series(10 * np.log10(ub_gem.mean(1))),
                                                        'y Train Avg':                  pd.Series(10 * np.log10([np.mean(y_train_avg)] * n_snr_pts)),
                                                        'y Test Avg':                   pd.Series(10 * np.log10([np.mean(y_test_avg)] * n_snr_pts))
                                                        }, axis=1)
                                results_df.to_csv(results_path + criterion + "_" + sigma_profile_type + "_" + data_type + "_bagging_" + bagging_method + ".csv")

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
                                        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
                                        ax1.set_xlabel('SNR [dB]', fontsize=18)
                                        ax1.set_ylabel(criterion.upper() + ' Reduction [%]', fontsize=18)
                                        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                                        ax2.set_ylabel(criterion.upper() + ' Degradation [%]',
                                                       fontsize=18)  # we already handled the x-label with ax1

                                        ref_err = np.min((err_rcln[-1, _m_idx, :].mean(0), err_cln[-1, _m_idx, :].mean(0)))
                                        gain = 100*(err_nr[:, _m_idx, :].mean(1) - err_r[:, _m_idx, :].mean(1)) / ref_err
                                        deg = 100*(err_r[:, _m_idx, :].mean(1) - ref_err) / ref_err
                                        ax1.plot(snr_db_vec, gain, '-ob')
                                        ax1.set_title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type + ", Model error=" + "{:.3f}".format(ref_err))
                                        ax2.plot(snr_db_vec, deg, '-xb')
                                        plt.show(block=False)
                                        for ii, xx in enumerate(snr_db_vec):
                                            ax1.annotate(
                                                "{:.2f}".format(err_nr[ii, _m_idx, :].mean()),
                                                xy=(xx, gain[ii]), xycoords='data',
                                                xytext=(0, 0), textcoords='offset points',
                                                fontsize=10)
                                            ax2.annotate(
                                                "{:.2f}".format(err_r[ii, _m_idx, :].mean()),
                                                xy=(xx, deg[ii]), xycoords='data',
                                                xytext=(0, 0), textcoords='offset points',
                                                fontsize=10)

                                        ax1.set_ylim(bottom=0)
                                        # ax2.tick_params(axis='y', labelcolor=color)
                                        # ax2.set_ylim(ax1.get_ylim())
                                        # We change the fontsize of minor ticks label
                                        ax1.tick_params(axis='both', which='major', labelsize=16)
                                        ax1.tick_params(axis='both', which='minor', labelsize=16)
                                        ax2.tick_params(axis='both', which='major', labelsize=16)
                                        ax2.tick_params(axis='both', which='minor', labelsize=16)
                                        fig.tight_layout()  # otherwise the right y-label is slightly clipped

                                # Plot error bounds for mae
                                if criterion == "mae":
                                    for _m_idx, _m in enumerate(ensemble_size):
                                        plt.figure(figsize=(12, 8))
                                        plt.plot(snr_db_vec, ub_bem.mean(1), '--g', marker='+', label='Upper bound')
                                        plt.plot(snr_db_vec, ub_gem.mean(1), '--g', marker='x', label='Upper bound')
                                        plt.plot(snr_db_vec, lb_1.mean(1), '-g', marker='+', label='Lower bound')
                                        plt.plot(snr_db_vec, lb_2.mean(1), '-g', marker='x', label='Lower bound')
                                        plt.title("dataset: " + str(data_type) + ", T=" + str(_m) + " regressors\nnoise=" + sigma_profile_type)
                                        plt.xlabel('SNR [dB]'), plt.ylabel(criterion.upper())
                                        plt.legend()
                                        plt.show(block=False)
                        print("---------------------------------------------------------------------------\n")

