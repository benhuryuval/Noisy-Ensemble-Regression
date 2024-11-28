import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)

# # # Robust vs non-robust BEM target plot
if True:
    import sklearn as sk
    import sklearn.ensemble
    import BaggingRobustRegressor
    import auxilliaryFunctions as aux

    data_type = "sin"
    n_samples = 1000
    test_size = 0.8
    train_noise = 0.1

    n_estimators = 10
    tree_max_depth = 6

    snr_db = -20
    snr = 10 ** (snr_db / 10)

    fig_nr, ax_nr = plt.figure("bem_example_nonrobust", figsize=(8, 6), dpi=300), plt.axes()
    fig_r, ax_r = plt.figure("bem_example_robust", figsize=(8, 6), dpi=300), plt.axes()
    fig_func_nr, ax_func_nr = plt.figure("bem_example_func_nonrobust", figsize=(8, 6), dpi=300), plt.axes()
    fig_func_r, ax_func_r = plt.figure("bem_example_func_robust", figsize=(8, 6), dpi=300), plt.axes()

    # Get data set
    X_train, y_train, X_test, y_test = aux.get_dataset(data_type=data_type, test_size=test_size, n_samples=n_samples,
                                                       noise=train_noise)
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Set prediction noise
    sig_var = np.var(y_train)
    sigma0 = sig_var / snr  # noise variance

    # - - - 90% corrupt predictors
    sigma_profile = np.zeros([n_estimators, 1])
    # set noise covariance
    sigma_profile[n_estimators - 1, :] = train_noise #sigma_profile[n_estimators - 1, :] / 100
    sigma_profile[0:n_estimators - 1, :] = (sigma0-sigma_profile.sum()) / (n_estimators - 2)
    noise_covariance = np.diag(sigma_profile.ravel())
    # Define regression models
    regr_1_bem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                                                   n_estimators=n_estimators, random_state=rng)
    regr_2_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'bem')
    regr_3_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'robust-bem')
    # Fit
    regr_1_bem.fit(X_train, y_train)
    regr_2_bem.fit(X_train, y_train)
    regr_3_bem.fit(X_train, y_train)
    # Predict
    y_1 = regr_1_bem.predict(X_test)
    y_2 = regr_2_bem.predict(X_test)
    y_3 = regr_3_bem.predict(X_test)
    # Plot
    ind_test = np.argsort(X_test[:, 0])
    ax_func_nr.plot(X_test[ind_test], y_2[ind_test], c="r", label="90%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    ax_func_r.plot(X_test[ind_test], y_3[ind_test], c="r", label="90%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    ax_nr.plot(y_test, y_2, c="r", label="90%", linestyle='', linewidth=2, marker='.',
               markersize=3)
    ax_r.plot(y_test, y_3, c="r", label="90%", linestyle='', linewidth=2, marker='.',
              markersize=3)
    print("90%: Non-robust: "+str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)))
    print("90%: Robust: "+str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)))

    # # - - - 50% corrupt predictors
    # sigma_profile = np.ones([n_estimators, 1])
    # # set noise covariance
    # sigma_profile[int(n_estimators/2):n_estimators, :] = train_noise #sigma_profile[int(n_estimators/2):n_estimators - 1, :] / 100
    # sigma_profile[0:int(n_estimators/2), :] = (sigma0-sigma_profile.sum()) / (int(n_estimators/2))
    # noise_covariance = np.diag(sigma_profile.ravel())
    # # Define regression models
    # regr_1_bem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
    #                                                n_estimators=n_estimators, random_state=rng)
    # regr_2_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'bem')
    # regr_3_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'robust-bem')
    # # Fit
    # regr_1_bem.fit(X_train, y_train)
    # regr_2_bem.fit(X_train, y_train)
    # regr_3_bem.fit(X_train, y_train)
    # # Predict
    # y_1 = regr_1_bem.predict(X_test)
    # y_2 = regr_2_bem.predict(X_test)
    # y_3 = regr_3_bem.predict(X_test)
    # # Plot
    # ind_test = np.argsort(X_test[:, 0])
    # ax_func_nr.plot(X_test[ind_test], y_2[ind_test], c="g", label="50%", linestyle='-', linewidth=2, marker='',
    #          markersize=2)
    # ax_func_r.plot(X_test[ind_test], y_3[ind_test], c="g", label="50%", linestyle='-', linewidth=2, marker='',
    #          markersize=2)
    # ax_nr.plot(y_test, y_2, c="g", label="50%", linestyle='', linewidth=2, marker='.',
    #          markersize=3)
    # ax_r.plot(y_test, y_3, c="g", label="50%", linestyle='', linewidth=2, marker='.',
    #          markersize=3)
    # print("50%: Non-robust: "+str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)))
    # print("50%: Robust: "+str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)))

    # - - - 10% corrupt predictors
    sigma_profile = np.ones([n_estimators, 1])
    # set noise covariance
    sigma_profile[1:n_estimators, :] = train_noise #sigma_profile[1:n_estimators-1, :] / 100
    sigma_profile[0, :] = (sigma0-sigma_profile.sum())
    noise_covariance = np.diag(sigma_profile.ravel())
    # Define regression models
    regr_1_bem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth),
                                                   n_estimators=n_estimators, random_state=rng)
    regr_2_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'bem')
    regr_3_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'robust-bem')
    # Fit
    regr_1_bem.fit(X_train, y_train)
    regr_2_bem.fit(X_train, y_train)
    regr_3_bem.fit(X_train, y_train)
    # Predict
    y_1 = regr_1_bem.predict(X_test)
    y_2 = regr_2_bem.predict(X_test)
    y_3 = regr_3_bem.predict(X_test)
    # Plot
    ind_test = np.argsort(X_test[:, 0])
    ax_func_nr.plot(X_test[ind_test], y_2[ind_test], c="b", label="10%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    ax_func_r.plot(X_test[ind_test], y_3[ind_test], c="b", label="10%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    ax_nr.plot(y_test, y_2, c="b", label="10%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    ax_r.plot(y_test, y_3, c="b", label="10%", linestyle='', linewidth=2, marker='.',
             markersize=3)
    print("10%: Non-robust: "+str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)))
    print("10%: Robust: "+str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)))

    # Plot training target
    ind_train = np.argsort(X_train[:, 0])
    ax_func_nr.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='', marker='o', markersize=4)
    ax_func_r.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='', marker='o', markersize=4)
    ax_nr.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', markersize=2)
    ax_r.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', markersize=2)

    # plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
    ax_func_nr.set_xlabel("x", fontsize=18)
    ax_func_nr.set_ylabel("Prediction", fontsize=18)
    ax_func_nr.legend(fontsize=18)
    ax_func_nr.tick_params(axis='x', labelsize=16)
    ax_func_nr.tick_params(axis='y', labelsize=16)
    ax_func_nr.set_ylim((-5, 5))

    ax_func_r.set_xlabel("x", fontsize=18)
    ax_func_r.set_ylabel("Prediction", fontsize=18)
    ax_func_r.legend(fontsize=18)
    ax_func_r.legend(fontsize=18)
    ax_func_r.tick_params(axis='x', labelsize=16)
    ax_func_r.tick_params(axis='y', labelsize=16)
    ax_func_r.set_ylim((-5, 5))

    ax_nr.set_xlabel("Prediction")
    ax_nr.set_ylabel("Target")
    ax_nr.legend(fontsize=18)

    ax_r.set_xlabel("Prediction", fontsize=18)
    ax_r.set_ylabel("Target", fontsize=18)
    ax_r.legend(fontsize=18)

    plt.show(block=False)

    fig_nr.savefig("nonrobust-bem-example-pred.png", dpi=300)
    fig_r.savefig("robust-bem-example-pred.png", dpi=300)
    fig_func_nr.savefig("nonrobust-bem-example-func.png", dpi=300)
    fig_func_r.savefig("robust-bem-example-func.png", dpi=300)
# # # # # # # # # # # # # # # # # # # # #

results_path = "Results//"
data_type_vec = ["kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]

# # # Robust vs non-robust MSE
if False:
    data_label = {
        "sin": "Sine",
        "exp": "Exp",
        "make_reg": "Linear",
        "kc_house_data": "King County",
        "diabetes": "Diabetes",
        "white-wine": "Wine"
    }

    for method_type in ["bem", "gem", "lr"]:
        fig = plt.figure(method_type+"_robust_vs_nonrobust", figsize=(8, 6), dpi=300)
        ax = plt.axes()
        plt.xlabel('SNR [dB]', fontsize=18)
        plt.ylabel('Test MSE Gain [dB]', fontsize=18)

        for data_type_idx, data_type in enumerate(data_type_vec):
            mse_results_df = pd.read_csv(results_path+data_type+"_"+method_type.lower()+".csv")
            snr_db_vec = mse_results_df["SNR"]
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(snr_db_vec,
                     10*np.log10(mse_results_df[method_type.upper()+', Prior'] / mse_results_df[method_type.upper()+', Robust']),
                     color=color, label=data_label[data_type], linestyle='', marker='o', markersize=2*(data_type_idx+1))
        ax.set_ylim(bottom=0)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # fig.set_size_inches(6.4, 4.8, forward=True)
        # fig.set_dpi(300)
        plt.show(block=False)
        fig.savefig(fig.get_label()+".png")
# # # # # # # # # # # # # # # # # # # # #

# # # Robust vs clean
if False:
    for method_type in ["bem", "gem", "lr"]:
        fig = plt.figure(method_type+"_robust_vs_clean")
        ax = plt.axes()
        plt.xlabel('SNR [dB]')
        plt.ylabel('Relative MSE (Test) [dB]')

        for data_type_idx, data_type in enumerate(data_type_vec):
            mse_results_df = pd.read_csv(results_path+data_type+"_"+method_type.lower()+".csv")
            snr_db_vec = mse_results_df["SNR"]
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(snr_db_vec, 10 * np.log10(mse_results_df[method_type.upper()+', Robust'] / mse_results_df[method_type.upper()+', Noiseless']),
                     color=color, label=data_type, linestyle='', marker='o', fillstyle='none', markersize=2*(data_type_idx+1))
        ax.set_ylim(bottom=0)
        plt.legend()
        plt.show(block=False)
# # # # # # # # # # # # # # # # # # # # #


if False:
    for method_type in ["bem", "gem", "lr"]:
        fig = plt.figure(method_type+"_robust_nonrobust_vs_clean")
        ax = plt.axes()
        plt.xlabel('SNR [dB]')
        plt.ylabel('Relative MSE (Test) [dB]')

        for data_type_idx, data_type in enumerate(data_type_vec):
            mse_results_df = pd.read_csv(results_path+data_type+"_"+method_type.lower()+".csv")
            snr_db_vec = mse_results_df["SNR"]
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(snr_db_vec, 10 * np.log10(mse_results_df[method_type.upper()+', Robust'] / mse_results_df[method_type.upper()+', Noiseless']),
                     color=color, label=data_type, linestyle='', marker='o', fillstyle='none', markersize=2*(data_type_idx+1))
            plt.plot(snr_db_vec, 10 * np.log10(mse_results_df[method_type.upper()+', Prior'] / mse_results_df[method_type.upper()+', Noiseless']),
                     color=color, label=data_type, linestyle='', marker='x', markersize=2*(data_type_idx+1))
        ax.set_ylim(bottom=0)
        # reorder legend and set legend column titles
        handles, labels = ax.get_legend_handles_labels()
        plot_handle = [plt.plot([], marker="", ls="")[0]] * 2
        handles = plot_handle[:1] + handles[::2] + plot_handle[:1] + handles[1::2]
        labels = ["Robust:"] + labels[::2] + ["Non-robust:"] + labels[1::2]
        # load legend
        plt.legend(handles, labels, ncol=2)
        plt.show(block=False)
