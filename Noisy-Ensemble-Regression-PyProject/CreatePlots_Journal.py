import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os  # from os.path import exists
import numpy as np

def mag2db(x, crit="mse"):
    if crit.upper() == "MSE":
        return 10*np.log10(x)
    else:
        return 10 * np.log10(x)
def db2mag(x, crit="mse"):
    if crit.upper() == "MSE":
        return 10**(x/10)
    else:
        return 10**(x/10)

data_type_vec = ["sin", "make_reg", "diabetes", "white-wine", "kc_house_data"]
T = 16

# results_folder_path = os.path.join("Results", "2024_04_03")  # MSE/MAE results for JSAC paper

# criterion, reg_algo, bagging_method, sigma_profile_type = "mse", "Bagging", "lr", "uniform"
# criterion, reg_algo, bagging_method, sigma_profile_type = "mse", "Bagging", "lr", "noiseless_even"

# criterion, reg_algo, bagging_method, sigma_profile_type = "mae", "Bagging", "gem", "uniform"
# criterion, reg_algo, bagging_method, sigma_profile_type = "mae", "Bagging", "gem", "noiseless_even"

# criterion, reg_algo, bagging_method, sigma_profile_type = "mse", "GradBoost", "gem", "uniform"
# criterion, reg_algo, bagging_method, sigma_profile_type = "mse", "GradBoost", "gem", "noiseless_even"

# criterion, reg_algo, bagging_method, sigma_profile_type = "mae", "GradBoost", "gem", "uniform"
# criterion, reg_algo, bagging_method, sigma_profile_type = "mae", "GradBoost", "gem", "noiseless_even"

# exp_name = "_".join((str(T), criterion, sigma_profile_type, reg_algo.lower(), bagging_method))



# results_folder_path = os.path.join("Results", "2024_01_12")  # MAE gains for JSAC paper
# criterion, reg_algo, bagging_method, sigma_profile_type = "mae", "Bagging", "gem", "noiseless_even"
# exp_name = "_".join((str(T), criterion, sigma_profile_type, reg_algo.lower(), bagging_method))



results_folder_path = os.path.join("Results", "2024_10_20")  # MSE with laplace noise
criterion, reg_algo, bagging_method, sigma_profile_type = "mse", "Bagging", "lr", "uniform"
exp_name = "_".join((criterion, sigma_profile_type, reg_algo.lower(), bagging_method))



results_path = os.path.join(results_folder_path, exp_name)

# # # # # # Robust vs non-robust MSE
data_label = {
    "sin": "Sine",
    "exp": "Exp",
    "make_reg": "Hyperplane",
    "kc_house_data": "King County",
    "diabetes": "Diabetes",
    "white-wine": "Wine"
}

figname = "_".join((exp_name, "Gain"))
figname_ = "_".join((exp_name, "RobustVsNonrobust"))
# plt.rcParams['text.usetex'] = True
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.set_label(figname)
ax1.set_xlabel('SNR [dB]', fontsize=18)
ax1.set_ylabel(criterion.upper()+' Reduction [%]', fontsize=18)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel(criterion.upper()+' Degradation [%]', fontsize=18)  # we already handled the x-label with ax1

fig_, ax_ = plt.subplots(1, 1, figsize=(12, 9))
fig_.set_label(figname_)
markers, labels = ["1", "2", "3", "4", "+"], data_type_vec
colors = []
for data_type_idx, data_type in enumerate(data_type_vec):
    if reg_algo == "Bagging":
        fname = "_".join((criterion, sigma_profile_type, data_type, reg_algo.lower(), bagging_method)) + ".csv"
        # fname = "_".join((str(T), criterion, sigma_profile_type, data_type, reg_algo.lower(), bagging_method)) + ".csv"
    elif reg_algo == "GradBoost":
        fname = "_".join((criterion, sigma_profile_type, data_type, "gbr")) + ".csv"
    path_to_file = os.path.join(results_path, fname)
    if os.path.exists(path_to_file):
        err_results_df = pd.read_csv(path_to_file)
    else:
        continue

    snr_db_vec = err_results_df["SNR"]
    err_nr, err_r, err_cln = err_results_df[reg_algo + ', Non-Robust'], err_results_df[reg_algo + ', Robust'], \
                             err_results_df[reg_algo + ', Noiseless']
    if True:
        err_reduction = (db2mag(err_nr) - db2mag(err_r)) / db2mag(err_cln)
        err_degradation = (db2mag(err_r) - db2mag(err_cln)) / db2mag(err_cln)

        color = next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(snr_db_vec, 100*err_reduction,
                 color=color, label=data_label[data_type], linestyle='', marker='o',
                 markersize=2*(len(data_type_vec)-data_type_idx))
        ax1.legend(fontsize=12)

        # ax2.plot(snr_db_vec, 100*err_degradation, color=color, label=data_label[data_type], linestyle='', marker='x',
        #          markersize=2*(len(data_type_vec)-data_type_idx))

        plt.show(block=False)

        # for ii, xx in enumerate(snr_db_vec):
        #     ax1.annotate(
        #         "{:.2f},{:.2f}".format(db2mag(err_r[ii]), db2mag(err_nr[ii])),
        #         xy=(xx, 100*err_reduction[ii]), xycoords='data',
        #         xytext=(0, 0), textcoords='offset points', size=4
        #     )

    colors.append(next(ax_._get_lines.prop_cycler)['color'])
    ax_.plot(snr_db_vec, 10**(err_nr/10), label=data_label[data_type], linestyle=':', marker=markers[data_type_idx], color=colors[data_type_idx], markersize=10, markeredgewidth=3)
    ax_.plot(snr_db_vec, 10**(err_r/10), label=data_label[data_type], linestyle='-', marker=markers[data_type_idx], color=colors[data_type_idx], markersize=10, markeredgewidth=3)
    # plt.plot(snr_db_vec, 10**(err_cln/10), '--k', label='Clean')
    # plt.title("dataset: " + str(data_type) + ", T=" + str(T) + " regressors\noise=" + sigma_profile_type)
    plt.legend()
    plt.xlim((-3, 10))
    plt.ylim((0, 1100))
    plt.grid()
    # plt.title(exp_name)
    plt.show(block=False)

ax1.set_ylim(bottom=0)
# We change the fontsize of minor ticks label
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(axis='both', which='minor', labelsize=16)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylim(ax1.get_ylim())
# ax2.tick_params(axis='both', which='major', labelsize=16)
# ax2.tick_params(axis='both', which='minor', labelsize=16)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax_.set_xlabel('SNR [dB]', fontsize=18), ax_.set_ylabel(criterion.upper(), fontsize=18)
ax_.tick_params(axis='both', which='major', labelsize=16), ax_.tick_params(axis='both', which='minor', labelsize=16)
# change legend order
handles, labels = ax_.get_legend_handles_labels()  # get handles and labels
idxs = np.linspace(0, 2 * (len(data_type_vec) - 1), len(data_type_vec), dtype=np.int32)
order = 1 + idxs  # np.concatenate((idxs, 1+idxs))  # specify order of items in legend
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=12, ncol=1)  # add legend to plot
#
def style_legend_titles_by_setting_position(leg: mpl.legend.Legend, bold: bool = False) -> None:
    """ Style legend "titles"

    A legend entry can be marked as a title by setting visible=False. Titles
    get left-aligned and optionally bolded.
    """
    # matplotlib.offsetbox.HPacker unconditionally adds a pixel of padding
    # around each child.
    hpacker_padding = 2

    for handle, label in zip(leg.legendHandles, leg.texts):
        if not handle.get_visible():
            # See matplotlib.legend.Legend._init_legend_box()
            widths = [leg.handlelength, leg.handletextpad]
            offset_points = sum(leg._fontsize * w for w in widths)
            offset_pixels = leg.figure.canvas.get_renderer().points_to_pixels(offset_points) + hpacker_padding
            label.set_position((-offset_pixels, 0))
            if bold:
                label.set_fontweight('bold')

def make_legend_with_subtitles(colors, data_type) -> mpl.legend.Legend:
    legend_contents = [
        (Patch(visible=False), 'Dataset'),
        (plt.Line2D([], [], linestyle=':', color=colors[0], marker=markers[0], markersize=10, markeredgewidth=3), data_type[0]),
        (plt.Line2D([], [], linestyle=':', color=colors[1], marker=markers[1], markersize=10, markeredgewidth=3), data_type[1]),
        (plt.Line2D([], [], linestyle=':', color=colors[2], marker=markers[2], markersize=10, markeredgewidth=3), data_type[2]),
        (plt.Line2D([], [], linestyle=':', color=colors[3], marker=markers[3], markersize=10, markeredgewidth=3), data_type[3]),
        (plt.Line2D([], [], linestyle=':', color=colors[4], marker=markers[4], markersize=10, markeredgewidth=3), data_type[4]),
        # (Patch(linestyle=':', color=colors[0]), data_type[0]),
        # (Patch(linestyle=':', color=colors[1]), data_type[1]),
        # (Patch(linestyle=':', color=colors[2]), data_type[2]),
        # (Patch(linestyle=':', color=colors[3]), data_type[3]),
        # (Patch(linestyle=':', color=colors[4]), data_type[4]),

        # (Patch(visible=False), ''),  # spacer

        (Patch(visible=False), 'Method'),
        (plt.Line2D([], [], linestyle=':', color='black'), 'Non-robust'),
        (plt.Line2D([], [], linestyle='-', color='black'), 'Robust (This work)'),
    ]
    fig = plt.figure(figsize=(2, 2))
    leg = fig.legend(*zip(*legend_contents))
    handles = [legend_contents[i][0] for i in range(9)]
    labels = [legend_contents[i][1] for i in range(9)]
    return leg, handles, labels
names = [data_label[data_type_vec[0]], data_label[data_type_vec[1]], data_label[data_type_vec[2]], data_label[data_type_vec[3]], data_label[data_type_vec[4]]]
leg, handles, labels = make_legend_with_subtitles(colors, names)
ax_.legend(handles=handles, labels=labels, fontsize=16)
style_legend_titles_by_setting_position(ax_.get_legend())
hpacker_padding = 2
for handle, label in zip(ax_.get_legend().legendHandles, ax_.get_legend().texts):
    if not handle.get_visible():
        # See matplotlib.legend.Legend._init_legend_box()
        widths = [ax_.get_legend().handlelength, ax_.get_legend().handletextpad]
        offset_points = sum(ax_.get_legend()._fontsize * w for w in widths)
        offset_pixels = ax_.get_legend().figure.canvas.get_renderer().points_to_pixels(offset_points) + hpacker_padding
        label.set_position((-offset_pixels, 0))
        if True:
            label.set_fontweight('bold')
ax_.grid()
ax_.set_ylim(bottom=0, top=1.5)
ax_.grid()


# fig.savefig(os.path.join(results_path, fig.get_label()+".png"))
# fig_.savefig(os.path.join(results_path, fig_.get_label()+".png"))
# fig.savefig(os.path.join(results_folder_path, fig.get_label()+".png"))
# fig_.savefig(os.path.join(results_folder_path, fig_.get_label()+".png"))

# # # # # # GB vs BAGGING
if False:
    fig, ax = plt.figure(criterion.upper() + "_" + reg_algo + "_" + "rGBR_vs_r" + bagging_method.upper() + "_" + sigma_profile_type + "_Gap", figsize=(8, 6), dpi=300), plt.axes()
    plt.xlabel('SNR [dB]', fontsize=18)
    plt.ylabel(criterion.upper()+' Gap [dB]', fontsize=18)

    for data_type_idx, data_type in enumerate(data_type_vec):
        fname_bag = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "bagging" + "_" + bagging_method + ".csv"
        err_results_df_bag = pd.read_csv(results_path + fname_bag)
        snr_db_vec = err_results_df_bag["SNR"]

        fname_gbr = str(T) + "_" + criterion + "_" + sigma_profile_type + "_" + data_type + "_" + "gbr" + ".csv"
        err_results_df_gbr = pd.read_csv(results_path + fname_gbr)

        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(snr_db_vec,
                 err_results_df_bag["Bagging"+', Non-Robust'] - err_results_df_gbr["GradBoost"+', Non-Robust'],
                 color=color, label=data_label[data_type], linestyle='', marker='o', markersize=2*(len(data_type_vec)-data_type_idx))
        plt.legend(fontsize=12)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show(block=False)

    # fig.savefig(results_path+fig.get_label()+".png")

# # # # # # Plot bounds for MAE, Bagging w\ normalized weights
if criterion.upper() == "MAE" and reg_algo == "Bagging":
    figname = exp_name + "_Bounds"
    fig, ax = plt.figure(figname, figsize=(12, 9)), plt.axes()
    plt.xlabel("SNR [dB]", fontsize=18)
    plt.ylabel("MAE [dB]", fontsize=18)

    for data_type_idx, data_type in enumerate(data_type_vec):
        fname = "_".join((str(T), criterion, sigma_profile_type, data_type, reg_algo.lower(), bagging_method)) + ".csv"
        path_to_file = os.path.join(results_path, fname)
        if os.path.exists(path_to_file):
            err_results_df = pd.read_csv(path_to_file)
        else:
            continue
        snr_db_vec = err_results_df["SNR"]

        # scaler = err_results_df[reg_algo+", Robust"].array[-1]
        scaler = err_results_df["y Train Avg"].array[0]
        color = next(ax._get_lines.prop_cycler)['color']
        # plt.plot(snr_db_vec, err_results_df['Lower bound (CLN), Robust'] - scaler, color=color, label=data_label[data_type], linestyle='--')
        # plt.plot(snr_db_vec, err_results_df['Lower bound (FLD), Robust'] - scaler, color=color, label=data_label[data_type], linestyle='--')
        plt.plot(snr_db_vec, 2*err_results_df[['Lower bound (CLN), Robust', 'Lower bound (FLD), Robust']].max(axis=1) - scaler, color=color, label=data_label[data_type], linestyle='--')
        # plt.plot(snr_db_vec, err_results_df['Upper bound (BEM), Robust']-scaler, color=color, label=data_label[data_type], linestyle='-', marker='o')
        # plt.plot(snr_db_vec, err_results_df['Upper bound (GEM), Robust']-scaler, color=color, label=data_label[data_type], linestyle='-', marker='x')
        plt.plot(snr_db_vec, 2*err_results_df[['Upper bound (BEM), Robust', 'Upper bound (GEM), Robust']].min(axis=1) - scaler, color=color, label=data_label[data_type], linestyle='-', marker='o')
        # plt.plot(snr_db_vec, err_results_df[reg_algo + ", Robust"]-scaler, color=color, label=data_label[data_type], linestyle=':')

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show(block=False)
        # plt.legend(fontsize=12, ncol=2)

    # change legend order
    handles, labels = ax.get_legend_handles_labels()  # get handles and labels
    idxs = np.linspace(0, 2*(len(data_type_vec)-1), len(data_type_vec), dtype=np.int32)
    order = 1+idxs  #np.concatenate((idxs, 1+idxs))  # specify order of items in legend
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=12, ncol=1)  # add legend to plot
    plt.grid()

    fig.savefig(os.path.join(results_path, fig.get_label()+".png"))
    # fig.savefig(os.path.join(results_folder_path, fig.get_label()+".png"))


# # # # # # # # # # # # # # # # # # # # #

