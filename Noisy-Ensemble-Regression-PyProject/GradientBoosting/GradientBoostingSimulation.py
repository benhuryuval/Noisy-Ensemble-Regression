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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rng = np.random.RandomState(42)
plt.rcParams.update({'font.size': 20})
cm = 1/2.54
fsiz = 20*cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings
plot_flag = False
results_path = "Results//"

n_repeat = 100  # Number of iterations for estimating expected performance
n_samples, test_size = 500, 0.5 # Size of the (synthetic) dataset / test data fraction
train_noise = 0.1  # Standard deviation of the measurement / training noise
max_depth = 3  # Maximal depth of decision tree
learning_rate = 0.1  # learning rate of gradient boostings
snr_db_vec = np.linspace(-40, 25, 10)

# xv, yv = np.meshgrid(x_name, y_name, sparse=False, indexing='ij')
# noise_covariance = np.exp(xv-yv)

####################################################
# Gradient Boosting
####################################################
if True:
    data_type_vec = ["auto-mpg"]  # ["auto-mpg", "kc_house_data", "diabetes", "white-wine", "sin", "exp", "make_reg"]

    for data_type in data_type_vec:

        # Load dataset
        # if data_type == "auto-mpg":
        #     data = pd.read_csv('..//..//Datasets//auto-mpg.csv')
        #     x_name, y_name = 'weight', 'mpg'
        #     n_samples = len(data)
        #     X, y = data[x_name].values.reshape(n_samples, 1), data[y_name].values.reshape(n_samples, 1)
        #     X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,
        #                                                                            y,
        #                                                                            test_size=test_size,
        #                                                                            random_state=rng)
        # else:
        #     X_train, y_train, X_test, y_test = aux.get_dataset(data_type=data_type, test_size=test_size,
        #                                                        n_samples=n_samples, noise=train_noise)
        #     X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
        #     x_name, y_name = 'x', 'y'

        X_train, y_train, X_test, y_test = aux.get_dataset(data_type=data_type, test_size=test_size,
                                                               n_samples=n_samples, noise=train_noise)
        X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        elif X_train.shape[1] == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        for idx_snr_db, snr_db in enumerate(snr_db_vec):
            # Set noise variance
            snr = 10 ** (snr_db / 10)
            sig_var = np.var(y_train)
            sigma0 = sig_var / snr  # noise variance

            # Plotting all the points
            if X_train.shape[1] == 1:
                plt.figure(figsize=(12, 8))
                plt.plot(X_train[:, 0], y_train[:, 0], 'ok', label='Train')
                plt.plot(X_test[:, 0], y_test[:, 0], 'xk', label='Test')

            # Defining the number of iterations
            _m_iterations = [0, 1, 10, 100]

            for _m in _m_iterations:  # iterate number of trees
                print("dataset "+data_type+":"+" "+str(_m)+" classifiers")

                # Setting noise covariance matrix
                sigma_profile = np.ones([_m + 1, ]) / np.maximum(_m, 1)
                sigma_profile[0:1] *= 0.01
                sigma_profile[1:] *= 10
                sigma_profile /= sigma_profile.sum()
                sigma_profile *= sigma0
                noise_covariance = np.diag(sigma_profile)

                mse = 0
                pred = np.zeros(len(y_test))
                for _n in range(0, n_repeat):
                    # Initiating the tree
                    rgb = RobustRegressionGB(
                        X = X_train,
                        y = y_train,
                        max_depth=max_depth,
                        min_sample_leaf=10,
                        learning_rate=learning_rate,
                        NoiseCov=noise_covariance
                    )
                    # Fitting on data
                    rgb.fit(X_train, y_train, m=_m)

                    # Predicting
                    # yhat = rgb.predict(rgb.d[rgb.x_vars].values[:,0])
                    y_test_hat = rgb.predict(X_test)

                    # Saving the predictions to the training set
                    # data['yhat'] = yhat
                    # mse += np.square(np.subtract(rgb.data[rgb.y_var].values, yhat)).mean()
                    # data['yhat'] = y_test_hat
                    mse += np.square(np.subtract(y_test[:,0], y_test_hat)).mean()
                    pred += y_test_hat

                mse /= n_repeat
                pred /= n_repeat

                print("dataset "+data_type+", T="+str(_m)+": "+"mse="+str(mse))

                if X_train.shape[1]==1:
                    plt.plot(X_test[:,0], pred, 'o', label=f't={_m}, mse={mse}')
                    plt.title('mpg vs weight')
                    plt.xlabel('weight')
                    plt.ylabel('mpg')
                    plt.legend()
                    plt.show()