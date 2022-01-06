import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import BaggingRobustRegressor

plt.rcParams.update({'font.size': 20})
cm = 1/2.54
fsiz = 20*cm

# Settings
n_repeat = 50  # Number of iterations for computing expectations
n_train = 100 # 200  # Size of the training set
n_test = 250  # Size of the test set
train_noise = 0.1 #0.05  # Standard deviation of the measurement / training noise
n_base_estimators = 25  # Ensemble size
max_depth = 3 # Maximal depth of decision tree
# np.random.seed(0)
data_type = 'sin'  # 'sin' / 'exp'

# Set noise covariance
# noise_covariance = np.eye(n_base_estimators)

sigma_profile = 30*np.ones([n_base_estimators, ])
sigma_profile[0:1] = 0.1
noise_covariance = np.diag(sigma_profile)

# xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# noise_covariance = np.exp(xv-yv)

if False:
    # Change this for exploring the bias-variance decomposition of other
    # estimators. This should work well for estimators with high variance (e.g.,
    # decision trees or KNN), but poorly for estimators with low variance (e.g.,
    # linear models).
    estimators = [
        # ("Decision Tree", DecisionTreeRegressor(max_depth=max_depth, random_state=rng))
        # ,
        # ("AdaBoost.R", BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth),
        #      n_estimators=32, random_state=rng))
        # ,
        # ("AdaBoost.R", BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth),
        #      n_estimators=128, random_state=rng))
        # ,
        ("Bagging", BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth),
             n_estimators=n_base_estimators))
        ,
        # ("Noisy Bagging (BEM)", BaggingRobustRegressor(
        #     BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
        #     noise_covariance, n_base_estimators, 'bem'))
        # ,
        # ("Noisy Bagging (GEM)", BaggingRobustRegressor(
        #     BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
        #     noise_covariance, n_base_estimators, 'gem'))
        # ,
        ("Noisy Bagging (LR)", BaggingRobustRegressor(
            BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
            noise_covariance, n_base_estimators, 'lr'))
        ,
        # ("Noisy Bagging (rBEM)", BaggingRobustRegressor(
        #     BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
        #     noise_covariance, n_base_estimators, 'robust-bem'))
        # ,
        # ("Noisy Bagging (rGEM)", BaggingRobustRegressor(
        #     BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
        #     noise_covariance, n_base_estimators, 'robust-gem'))
        # ,
        ("Noisy Bagging (rLR)", BaggingRobustRegressor(
            BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
            noise_covariance, n_base_estimators, 'robust-lr'))
        # ,
        # ("NoisyWeightedBagging(un)", BaggingRobustRegressor(
        #     BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n_base_estimators),
        #     noise_covariance, n_base_estimators, 'robust-unconstrained'))
    ]

    n_estimators = len(estimators)

    # Generate data
    def f(x, type):
        x = x.ravel()
        if type == 'exp':
            return np.exp(-(x ** 2)) + 1.5 * np.exp(-((x - 2) ** 2))
        elif type == 'sin':
            return np.sin(x) + np.sin(3 * x)
        else:
            return np.zeros([len(x),])

    def generate(n_samples, noise, n_repeat=1, type=None):
        X = np.random.rand(n_samples) * 10 - 5
        X = np.sort(X)

        if n_repeat == 1:
            y = f(X, type) + np.random.normal(0.0, noise, n_samples)
        else:
            y = np.zeros((n_samples, n_repeat))

            for i in range(n_repeat):
                y[:, i] = f(X, type) + np.random.normal(0.0, noise, n_samples)

        X = X.reshape((n_samples, 1))

        return X, y


    X_train = []
    y_train = []

    for i in range(n_repeat):
        X, y = generate(n_samples=n_train, noise=train_noise, type=data_type)
        X_train.append(X)
        y_train.append(y)

    X_test, y_test = generate(n_samples=n_test, noise=train_noise, n_repeat=n_repeat, type=data_type)

    plt.figure(figsize=(10, 8))

    # Loop over estimators to compare
    for n, (name, estimator) in enumerate(estimators):
        # Compute predictions
        y_predict = np.zeros((n_test, n_repeat))

        for i in range(n_repeat):
            estimator.fit(X_train[i], y_train[i])
            y_predict[:, i] = estimator.predict(X_test)

        # Bias^2 + Variance + Noise decomposition of the mean squared error
        y_error = np.zeros(n_test)

        for i in range(n_repeat):
            for j in range(n_repeat):
                y_error += (y_test[:, j] - y_predict[:, i]) ** 2

        y_error /= n_repeat * n_repeat

        y_noise = np.var(y_test, axis=1)
        y_bias = (f(X_test, data_type) - np.mean(y_predict, axis=1)) ** 2
        y_var = np.var(y_predict, axis=1)

        print(
            "{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
            " + {3:.4f} (var) + {4:.4f} (noise)".format(
                name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)
            )
        )

        # Plot figures
        plt.subplot(2, n_estimators, n + 1)
        # for i in range(n_repeat):
        #     if i == 0:
        #         plt.plot(X_test, y_predict[:, i], "r", label=r"$\^y(x)$")
        #     else:
        #         plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

        plt.plot(X_test, f(X_test, data_type), "b", label="$f(x)$")
        plt.plot(X_test, np.mean(y_predict, axis=1), "--c", label=r"$\mathbb{E}_{LS} \^y(x)$")
        plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

        plt.xlim([-5, 5])
        plt.title(name)

        if n == n_estimators - 1:
            plt.legend(loc=(1.1, 0.5))

        plt.subplot(2, n_estimators, n_estimators + n + 1)
        plt.plot(X_test, 10*np.log10(y_error), "r", label="$error(x)$")
        plt.plot(X_test, 10*np.log10(y_bias), "b", label="$bias^2(x)$")
        plt.plot(X_test, 10*np.log10(y_var), "g", label="$variance(x)$")
        plt.plot(X_test, 10*np.log10(y_noise), "c", label="$noise(x)$")

        plt.xlim([-5, 5])
        plt.ylim([-100, 10])
        # plt.ylim([0, 3])

        if n == n_estimators - 1:
            plt.legend(loc=(1.1, 0.5))

        ########
        # plt.figure(figsize=(10, 8))
        # plt.plot(X_test, f(X_test, data_type), "b", label="$f(x)$")
        # plt.plot(X_test, np.mean(y_predict, axis=1), "--c", label=r"$\mathbb{E}_{LS} \^y(x)$")
        # plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")
        # plt.xlim([-5, 5])
        # plt.title(name)


    plt.subplots_adjust(right=0.75)
    plt.show()

####################################################
# Clean Bagging with T=1, 8, 512
####################################################
if True:
    # Create the dataset
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=4)

    # regr_2 = AdaBoostRegressor(
    #     DecisionTreeRegressor(max_depth=4), n_estimators=8, random_state=rng
    # )
    # regr_3 = AdaBoostRegressor(
    #     DecisionTreeRegressor(max_depth=4), n_estimators=256, random_state=rng
    # )

    regr_2 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=8, random_state=rng)
    regr_3 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=512, random_state=rng)

    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)

    # Plot the results
    plt.figure(figsize=(fsiz,fsiz))
    plt.scatter(X, y, c="k", label="Training set")
    plt.plot(X, y_1, c="g", label="T=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="T=8", linewidth=2)
    plt.plot(X, y_3, c="b", label="T=512", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    # plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show(block=False)
    plt.savefig('bagging_example.png')

####################################################
# Noisy Bagging using BEM and GEM with T=8, 32, non-uniform \sigma
####################################################
if True:
    # Create the dataset
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
    sigma0 = 10
    sigma1 = sigma0/100

    # # # BEM
    # Fit regression model
    regr_1 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=8, random_state=rng)

    sigma_profile = sigma0*np.ones([8, ])
    sigma_profile[1::2] = sigma1
    noise_covariance = np.diag(sigma_profile)
    regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_1, noise_covariance, 8, 'bem')

    regr_3 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=32, random_state=rng)

    sigma_profile = sigma0*np.ones([32, ])
    sigma_profile[1::2] = sigma1
    noise_covariance = np.diag(sigma_profile)
    regr_4 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_3, noise_covariance, 32, 'bem')

    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)
    y_4 = regr_4.predict(X)

    # Plot the results
    plt.figure(figsize=(fsiz,fsiz))
    plt.scatter(X, y, c="k", label="Training set")
    # plt.plot(X, y_1, c="r", ls="-.", label="T=8, \sigma=0", linewidth=2)
    plt.plot(X, y_2, c="r", label="T=8", linewidth=2)
    # plt.plot(X, y_3, c="b", ls="-.", label="T=512, \sigma=0", linewidth=2)
    plt.plot(X, y_4, c="b", label="T=32", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    # plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show(block=False)
    plt.savefig('noisy_prediction_example.png')

    # # # # GEM
    # # Fit regression model
    # regr_1 = BaggingRegressor(
    #     DecisionTreeRegressor(max_depth=4), n_estimators=8, random_state=rng)
    #
    # sigma_profile = sigma0*np.ones([8, ])
    # sigma_profile[1::2] = sigma1
    # noise_covariance = np.diag(sigma_profile)
    # regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(
    #         regr_1, noise_covariance, 8, 'gem')
    #
    # regr_3 = BaggingRegressor(
    #     DecisionTreeRegressor(max_depth=4), n_estimators=32, random_state=rng)
    #
    # sigma_profile = sigma0*np.ones([32, ])
    # sigma_profile[1::2] = sigma1
    # noise_covariance = np.diag(sigma_profile)
    # regr_4 = BaggingRobustRegressor.BaggingRobustRegressor(
    #         regr_3, noise_covariance, 32, 'gem')
    #
    # regr_1.fit(X, y)
    # regr_2.fit(X, y)
    # regr_3.fit(X, y)
    # regr_4.fit(X, y)
    #
    # # Predict
    # y_1 = regr_1.predict(X)
    # y_2 = regr_2.predict(X)
    # y_3 = regr_3.predict(X)
    # y_4 = regr_4.predict(X)
    #
    # # Plot the results
    # plt.figure()
    # plt.scatter(X, y, c="k", label="Training set")
    # # plt.plot(X, y_1, c="r", ls="-.", label="T=8, \sigma=0", linewidth=2)
    # plt.plot(X, y_2, c="r", label="T=8", linewidth=2)
    # # plt.plot(X, y_3, c="b", ls="-.", label="T=512, \sigma=0", linewidth=2)
    # plt.plot(X, y_4, c="b", label="T=32", linewidth=2)
    # plt.xlabel("Data")
    # plt.ylabel("Target")
    # # plt.title("Boosted Decision Tree Regression")
    # plt.legend()
    # plt.show()
    # # plt.ylim([-4,4])

####################################################
# Noisy Bagging using BEM, rBEM and GEM, rGEM with T=8, 512, non-uniform \sigma
####################################################
if True:
    # Create the dataset
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    f = np.sin(X).ravel() + np.sin(6 * X).ravel()
    y = f + rng.normal(0, 0.1, X.shape[0])

    T = 8
    sigma0 = 10
    sigma1 = sigma0/100

    sigma_profile = sigma0*np.ones([T, ])
    sigma_profile[1::2] = sigma1
    noise_covariance = np.diag(sigma_profile)

    # # # BEM
    # Fit regression model
    regr_1 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_1, noise_covariance, T, 'bem')

    regr_3 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_4 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_3, noise_covariance, T, 'robust-bem')

    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)
    y_4 = regr_4.predict(X)

    # Plot the results
    plt.figure(figsize=(fsiz,fsiz))
    plt.plot(X, f, c="k", label="Target", linewidth=2)
    plt.scatter(X, y, c="k", label="Training set")
    # plt.plot(X, y_1, c="r", ls="-.", label="T=8, \sigma=0", linewidth=2)
    plt.plot(X, y_2, c="r", label="T="+str(T)+", BEM", linewidth=2)
    # plt.plot(X, y_3, c="b", ls="-.", label="T=512, \sigma=0", linewidth=2)
    plt.plot(X, y_4, c="b", label="T="+str(T)+", rBEM", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    # plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show(block=False)
    plt.savefig('r_bem_example.png')

    print("BEM:")
    print("\tTrain MSE: \n\tNoiseless:" + str(np.mean((y_1 - y) ** 2))+"\n\tPrior: " + str(np.mean((y_2 - y) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - y) ** 2)))
    print("\tTest MSE: \n\tNoiseless:" + str(np.mean((y_1 - f) ** 2)) + "\n\tPrior: " + str(np.mean((y_2 - f) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - f) ** 2)))

    # # # GEM
    # Fit regression model
    regr_1 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_1, noise_covariance, T, 'gem')

    regr_3 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_4 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_3, noise_covariance, T, 'robust-gem')

    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)
    y_4 = regr_4.predict(X)

    # Plot the results
    plt.figure(figsize=(fsiz,fsiz))
    plt.plot(X, f, c="k", label="Target", linewidth=2)
    plt.scatter(X, y, c="k", label="Training set")
    # plt.plot(X, y_1, c="r", ls="-.", label="T=8, \sigma=0", linewidth=2)
    plt.plot(X, y_2, c="r", label="T="+str(T)+", GEM", linewidth=2)
    # plt.plot(X, y_3, c="b", ls="-.", label="T=512, \sigma=0", linewidth=2)
    plt.plot(X, y_4, c="b", label="T="+str(T)+", rGEM", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    # plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show(block=False)
    plt.savefig('r_gem_example.png')

    print("GEM:")
    print("\tTrain MSE: \n\tNoiseless:" + str(np.mean((y_1 - y) ** 2))+"\n\tPrior: " + str(np.mean((y_2 - y) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - y) ** 2)))
    print("\tTest MSE: \n\tNoiseless:" + str(np.mean((y_1 - f) ** 2)) + "\n\tPrior: " + str(np.mean((y_2 - f) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - f) ** 2)))

    # # # LR
    # Fit regression model
    regr_1 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_1, noise_covariance, T, 'lr')

    regr_3 = BaggingRegressor(
        DecisionTreeRegressor(max_depth=4), n_estimators=T, random_state=rng)
    regr_4 = BaggingRobustRegressor.BaggingRobustRegressor(
            regr_3, noise_covariance, T, 'robust-lr')

    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)
    regr_4.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    y_3 = regr_3.predict(X)
    y_4 = regr_4.predict(X)

    # Plot the results
    plt.figure(figsize=(fsiz,fsiz))
    plt.plot(X, f, c="k", label="Target", linewidth=2)
    plt.scatter(X, y, c="k", label="Training set")
    # plt.plot(X, y_1, c="r", ls="-.", label="T=8, \sigma=0", linewidth=2)
    plt.plot(X, y_2, c="r", label="T="+str(T)+", LR", linewidth=2)
    # plt.plot(X, y_3, c="b", ls="-.", label="T=512, \sigma=0", linewidth=2)
    plt.plot(X, y_4, c="b", label="T="+str(T)+", rLR", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    # plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show(block=False)
    plt.savefig('r_lr_example.png')

    print("LR:")
    print("\tTrain MSE: \n\tNoiseless:" + str(np.mean((y_1 - y) ** 2))+"\n\tPrior: " + str(np.mean((y_2 - y) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - y) ** 2)))
    print("\tTest MSE: \n\tNoiseless:" + str(np.mean((y_1 - f) ** 2)) + "\n\tPrior: " + str(np.mean((y_2 - f) ** 2)) + "\n\tRobust: " + str(np.mean((y_4 - f) ** 2)))
