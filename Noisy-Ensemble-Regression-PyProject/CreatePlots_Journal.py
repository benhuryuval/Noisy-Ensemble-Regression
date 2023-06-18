import pandas as pd
import matplotlib.pyplot as plt
import os  # from os.path import exists
import numpy as np

data_type_vec = ["sin", "exp", "make_reg", "diabetes", "white-wine", "kc_house_data"]
# data_type_vec = ["sin", "exp", "make_reg", "diabetes", "white-wine"]
criterion = "mae"  # "mse" / "mae"
reg_algo = "GradBoost"  # "GradBoost" / "Bagging"
bagging_method = "bem"  # "bem" / "gem"
sigma_profile_type = "noiseless_even"  # "noiseless_even" / "uniform"
T = 16

results_path = "Results//2023_06_14//" + str(T) + "_" + criterion + "_" + sigma_profile_type + "//"

# # # Robust vs non-robust MSE
data_label = {
    "sin": "Sine",
    "exp": "Exp",
    "make_reg": "Linear",
    "kc_house_data": "King County",
    "diabetes": "Diabetes",
    "white-wine": "Wine"
}

if reg_algo == "Bagging":
    figname = criterion.upper() + "_" + reg_algo + "_" + bagging_method.upper() + "_" + sigma_profile_type + "_RobustVsNonrobust"
elif reg_algo == "GradBoost":
    figname = criterion.upper() + "_" + reg_algo + "_" + sigma_profile_type + "_RobustVsNonrobust"
fig, ax = plt.figure(figname,
                     figsize=(8, 6), dpi=300), plt.axes()
plt.xlabel('SNR [dB]', fontsize=18)
plt.ylabel(criterion.upper()+' Gain [dB]', fontsize=18)

for data_type_idx, data_type in enumerate(data_type_vec):
    if reg_algo == "Bagging":
        fname = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "bagging" + "_" + bagging_method + ".csv"
    elif reg_algo == "GradBoost":
        fname = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "gbr" + ".csv"
    path_to_file = results_path + fname
    if os.path.exists(path_to_file):
        err_results_df = pd.read_csv(path_to_file)
    else:
        continue
    snr_db_vec = err_results_df["SNR"]

    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(snr_db_vec,
             err_results_df[reg_algo+', Non-Robust'] - err_results_df[reg_algo+', Robust'],
             color=color, label=data_label[data_type], linestyle='', marker='o', markersize=2*(len(data_type_vec)-data_type_idx))
    # ax.set_ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # fig.set_size_inches(6.4, 4.8, forward=True)
    # fig.set_dpi(300)
    plt.show(block=False)
    # fig.savefig(fig.get_label()+".png")

    # plt.figure(figsize=(12, 8))
    # plt.plot(snr_db_vec, err_results_df[reg_algo+', Non-Robust'], '-xr', label='Non-robust')
    # plt.plot(snr_db_vec, err_results_df[reg_algo+', Robust'], '-ob', label='Robust')
    # plt.title("dataset: " + str(data_type) + ", T=" + str(T) + " regressors\nnoise=" + sigma_profile_type)
    # plt.xlabel('SNR [dB]')
    # plt.ylabel(criterion.upper() + ' [dB]')
    # plt.legend()
    # plt.show(block=False)
fig.savefig(results_path+fig.get_label()+".png")

if False:
    fig, ax = plt.figure(criterion.upper() + "_" + reg_algo + "_" + "rGBR_vs_r" + bagging_method.upper() + "_" + sigma_profile_type + "_Gap", figsize=(8, 6), dpi=300), plt.axes()
    plt.xlabel('SNR [dB]', fontsize=18)
    plt.ylabel(criterion.upper()+' Gap [dB]', fontsize=18)

    for data_type_idx, data_type in enumerate(data_type_vec):
        fname_bag = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "bagging" + "_" + bagging_method + ".csv"
        err_results_df_bag = pd.read_csv(results_path + fname_bag)
        snr_db_vec = err_results_df["SNR"]

        fname_gbr = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "gbr" + ".csv"
        err_results_df_gbr = pd.read_csv(results_path + fname_gbr)

        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(snr_db_vec,
                 err_results_df_bag["Bagging"+', Robust'] - err_results_df_gbr["GradBoost"+', Robust'],
                 color=color, label=data_label[data_type], linestyle='', marker='o', markersize=2*(len(data_type_vec)-data_type_idx))
        plt.legend(fontsize=12)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show(block=False)

    fig.savefig(results_path+fig.get_label()+".png")

# # # # # # # # # # # # # # # # # # # # #

