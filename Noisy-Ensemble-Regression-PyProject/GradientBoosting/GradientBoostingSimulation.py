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

plt.rcParams.update({'font.size': 20})
cm = 1/2.54
fsiz = 20*cm

# Settings
n_repeat = 250  # Number of iterations for computing expectations
n_base_estimators = 25  # Ensemble size

# n_train = 100 # 200  # Size of the training set
# n_test = 250  # Size of the test set
# train_noise = 0.1 #0.05  # Standard deviation of the measurement / training noise
# max_depth = 3 # Maximal depth of decision tree
# # np.random.seed(0)
# data_type = 'sin'  # 'sin' / 'exp'

# Set noise covariance
sigma_profile = 10*np.ones([n_base_estimators, ])
# sigma_profile[0:1] = 0.1
noise_covariance = np.diag(sigma_profile)

# xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# noise_covariance = np.exp(xv-yv)

####################################################
# Gradient Boosting
####################################################
if True:
    # Load dataset
    d = pd.read_csv('..//..//Datasets//auto-mpg.csv')
    y = 'mpg'
    x = 'weight'

    # Plotting all the points
    plt.figure(figsize=(12, 8))
    plt.plot(d[x], d[y], 'o', label='original')

    # Defining the number of iterations
    _m_iterations = [0, 1, 10, 100]

    for _m in _m_iterations:  # iterate number of trees

        # Setting noise covariance matrix
        sigma_profile = np.ones([_m + 1, ]) / np.maximum(_m, 1)
        sigma_profile[0:1] *= 0.01
        sigma_profile[1:] *= 100
        noise_covariance = np.diag(sigma_profile)

        mse = 0
        for _n in range(0, n_repeat):

            # Initiating the tree
            rgb = RobustRegressionGB(
                d,
                y,
                [x],
                max_depth=3,
                min_sample_leaf=10,
                learning_rate=0.1,
                NoiseCov=noise_covariance
            )
            # Fitting on data
            rgb.fit(m=_m)

            # Predicting
            yhat = rgb.predict(rgb.d[rgb.x_vars].values[:,0])

            # Saving the predictions to the training set
            d['yhat'] = yhat
            mse += np.square(np.subtract(rgb.d[rgb.y_var].values, yhat)).mean()
        mse /= n_repeat

        plt.plot(d[x], d['yhat'], 'o', label=f't={_m}, mse={mse}')

    plt.title('mpg vs weight')
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    plt.show()