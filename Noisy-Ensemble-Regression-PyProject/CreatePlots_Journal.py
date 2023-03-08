import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_path = "Results//2023_04_03//"
data_type_vec = ["sin", "exp", "make_reg", "kc_house_data", "diabetes", "white-wine"]
criterion = "mae"  # "mse" / "mae"
reg_algo = "Bagging"  # "GradBoost" / "Bagging"
bagging_method = "bem"  # "bem" / "gem"
sigma_profile_type = "noiseless_fraction"  # "noiseless_even" / "noiseless_fraction" / "uniform"
T = 16

# # # Robust vs non-robust MSE
data_label = {
    "sin": "Sine",
    "exp": "Exp",
    "make_reg": "Linear",
    "kc_house_data": "King County",
    "diabetes": "Diabetes",
    "white-wine": "Wine"
}

fig, ax = plt.figure(reg_algo + ", " + bagging_method + ": Robust vs Non-robust", figsize=(8, 6), dpi=300), plt.axes()
plt.xlabel('SNR [dB]', fontsize=18)
plt.ylabel(criterion.upper()+' Gain [dB]', fontsize=18)

for data_type_idx, data_type in enumerate(data_type_vec):
    if reg_algo == "Bagging":
        fname = data_type + "_" + "bagging" + "_" + bagging_method + "_" + str(T) + "_" + criterion + "_" + sigma_profile_type + ".csv"
    elif reg_algo == "GradBoost":
        fname = data_type + "_" + "gbr" + "_" + str(T) + "_" + criterion + "_" + sigma_profile_type + ".csv"

    err_results_df = pd.read_csv(results_path + fname)
    snr_db_vec = err_results_df["SNR"]

    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(snr_db_vec,
             err_results_df[reg_algo+', Non-Robust'] - err_results_df[reg_algo+', Robust'],
             color=color, label=data_label[data_type], linestyle='', marker='o', markersize=2*(len(data_type_vec)-data_type_idx))
    # ax.set_ylim(bottom=0)
    plt.legend(fontsize=18)
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

# # # # # # # # # # # # # # # # # # # # #
