# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.ensemble
import sklearn.datasets
import pandas as pd

import BaggingRobustRegressor
import auxilliaryFunctions as aux

rng = np.random.RandomState(42)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plot_flag = False
results_path = "..//Results//"
# data_type = 'white-wine'  # kc_house_data / diabetes / white-wine / sin / exp / make_reg
data_type_vec = ["kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]
# / carbon_nanotubes (cant use, 3dim target y)

n_samples = 1000
test_size = 0.8
train_noise = 0.1

n_estimators = 10
tree_max_depth = 6
snr_db_vec = np.linspace(-40, 25, 10)
# snr_db_vec = [-10]

for data_type in data_type_vec:
        print("Data set: "+data_type)
        mse_results_bem, mse_results_gem, mse_results_lr = [], [], []

        for idx_snr_db, snr_db in enumerate(snr_db_vec):
                snr = 10**(snr_db/10)

                # Get data set
                X_train, y_train, X_test, y_test = aux.get_dataset(data_type=data_type, test_size=test_size, n_samples=n_samples, noise=train_noise)
                X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

                # Set prediction noise
                sig_var = np.var(y_train)
                sigma0 = sig_var/snr  # noise variance

                sigma_profile = np.ones([n_estimators, 1])
                # sigma_profile[1:n_estimators-1, :] = 0 #sigma_profile[1:9, :] / 100
                # sigma_profile[5:n_estimators-1, :] = 0 #sigma_profile[1::2, :] / 100
                # sigma_profile[n_estimators-1, :] = 0 #sigma_profile[1::2, :] / 100
                # sigma_profile[1:n_estimators-1, :] = sigma_profile[1:n_estimators-1, :] / 100
                sigma_profile[5:n_estimators-1, :] = sigma_profile[5:n_estimators-1, :] / 100
                # sigma_profile[n_estimators-1, :] = sigma_profile[n_estimators-1, :] / 100
                sigma_profile /= sigma_profile.sum()
                sigma_profile *= sigma0
                noise_covariance = np.diag(sigma_profile.ravel())

                # - - - BEM
                # Fit regression model
                regr_1_bem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                regr_2_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'bem')
                regr_3_bem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_bem, noise_covariance, n_estimators, 'robust-bem')

                regr_1_bem.fit(X_train, y_train)
                regr_2_bem.fit(X_train, y_train)
                regr_3_bem.fit(X_train, y_train)

                # Predict
                y_1 = regr_1_bem.predict(X_test)
                y_2 = regr_2_bem.predict(X_test)
                y_3 = regr_3_bem.predict(X_test)

                # ind_train = np.argsort(X_train[:, 0])
                # ind_test = np.argsort(X_test[:, 0])
                # plt.figure()
                # plt.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='', marker='o',
                #          markersize=2)
                # plt.plot(X_test[ind_test], y_2[ind_test], c="b", label="Prior", linestyle='', linewidth=2, Marker='.',
                #          markersize=2)
                # plt.plot(X_test[ind_test], y_3[ind_test], c="g", label="Robust", linestyle='-', linewidth=2)
                # plt.xlabel("Data")
                # plt.ylabel("Target")
                # plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
                # plt.legend()
                # plt.show(block=False)

                if plot_flag:
                        # Plot model accuracy (no noise)
                        plt.figure()
                        plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                        plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='o', markersize=1)

                        # Plot the results
                        if X_test.shape[1] <= 1:
                                ind_train = np.argsort(X_train[:,0])
                                ind_test = np.argsort(X_test[:,0])
                                plt.figure()
                                # plt.plot(X_test[ind], y_test[ind], c="k", label="Target", linestyle='-', linewidth=1)
                                plt.plot(X_train[ind_train], y_train[ind_train], c="k", label="Training", linestyle='-', marker='o', markersize=1)
                                # plt.plot(X_test[ind], y_1[ind], c="r", label="Clean", linestyle='--', linewidth=2)
                                plt.plot(X_test[ind_test], y_2[ind_test], c="b", label="Prior", linestyle='', linewidth=2, Marker='.', markersize=1.5)
                                plt.plot(X_test[ind_test], y_3[ind_test], c="g", label="Robust", linestyle=':', linewidth=2)
                                plt.xlabel("Data")
                                plt.ylabel("Target")
                                plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
                                plt.legend()
                                plt.show(block=False)

                        plt.figure()
                        plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                        # plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='x', markersize=3)
                        plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                        plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                        plt.xlabel('Test target')
                        plt.ylabel('Test prediction')
                        plt.title('BEM')
                        plt.grid()
                        plt.legend()

                print("BEM MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                      )
                mse_results_bem.append([ snr_db,
                                        sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

                # - - - GEM
                # Fit regression model
                regr_1_gem = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                regr_2_gem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_gem, noise_covariance, n_estimators, 'gem')
                regr_3_gem = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_gem, noise_covariance, n_estimators, 'robust-gem')

                regr_1_gem.fit(X_train, y_train)
                regr_2_gem.fit(X_train, y_train)
                regr_3_gem.fit(X_train, y_train)

                # Predict
                y_1 = regr_1_gem.predict(X_test)
                y_2 = regr_2_gem.predict(X_test)
                y_3 = regr_3_gem.predict(X_test)

                # Plot the results
                if plot_flag:
                        plt.figure()
                        plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                        plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='.', markersize=2)
                        plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                        plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                        plt.xlabel('Test target')
                        plt.ylabel('Test prediction')
                        plt.title('GEM')
                        plt.grid()
                        plt.legend()

                print("GEM MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                      )
                mse_results_gem.append([ snr_db,
                                        sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

                # - - - LR
                # Fit regression model
                regr_1_lr = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=tree_max_depth), n_estimators=n_estimators, random_state=rng)
                regr_2_lr = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_lr, noise_covariance, n_estimators, 'lr')
                regr_3_lr = BaggingRobustRegressor.BaggingRobustRegressor(regr_1_lr, noise_covariance, n_estimators, 'robust-lr')

                regr_1_lr.fit(X_train, y_train)
                regr_2_lr.fit(X_train, y_train)
                regr_3_lr.fit(X_train, y_train)

                # Predict
                y_1 = regr_1_lr.predict(X_test)
                y_2 = regr_2_lr.predict(X_test)
                y_3 = regr_3_lr.predict(X_test)

                # Plot the results
                if plot_flag:
                        plt.figure()
                        plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
                        plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='.', markersize=2)
                        plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=2)
                        plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=2)
                        plt.xlabel('Test target')
                        plt.ylabel('Test prediction')
                        plt.title('LR')
                        plt.grid()
                        plt.legend()

                print("LR MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                                "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                                "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
                      )
                mse_results_lr.append([ snr_db,
                                        sk.metrics.mean_squared_error(y_test, y_1, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_2, squared=False),
                                        sk.metrics.mean_squared_error(y_test, y_3, squared=False)])

        mse_results_bem_df = pd.DataFrame(mse_results_bem, columns=['SNR', 'BEM, Noiseless', 'BEM, Prior', 'BEM, Robust'])
        mse_results_gem_df = pd.DataFrame(mse_results_gem, columns=['SNR', 'GEM, Noiseless', 'GEM, Prior', 'GEM, Robust'])
        mse_results_lr_df = pd.DataFrame(mse_results_lr, columns=['SNR', 'LR, Noiseless', 'LR, Prior', 'LR, Robust'])

        mse_results_bem_df.to_csv(results_path+data_type+"_bem.csv")
        mse_results_gem_df.to_csv(results_path+data_type+"_gem.csv")
        mse_results_lr_df.to_csv(results_path+data_type+"_lr.csv")

if plot_flag:
        plt.figure()
        plt.plot(snr_db_vec, 10*np.log10(mse_results_bem_df['BEM, Prior']/mse_results_bem_df['BEM, Noiseless']), c="k", label="BEM", linestyle='-', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_bem_df['BEM, Robust']/mse_results_bem_df['BEM, Noiseless']), c="k", label="rBEM", linestyle='-', marker='*', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_gem_df['GEM, Prior']/mse_results_gem_df['GEM, Noiseless']), c="r", label="GEM", linestyle='--', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_gem_df['GEM, Robust']/mse_results_gem_df['GEM, Noiseless']), c="r", label="rGEM", linestyle='--', marker='*', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_lr_df['LR, Prior']/mse_results_lr_df['LR, Noiseless']), c="b", label="LR", linestyle='-.', marker='x', linewidth=0.75)
        plt.plot(snr_db_vec, 10*np.log10(mse_results_lr_df['LR, Robust']/mse_results_lr_df['LR, Noiseless']), c="b", label="rLR", linestyle='-.', marker='*', linewidth=0.75)
        plt.xlabel('SNR [dB]')
        plt.ylabel('Relative MSE (Test) [dB]')
        # plt.title('')
        plt.grid()
        plt.legend()
