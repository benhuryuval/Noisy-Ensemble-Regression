# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.ensemble
import sklearn.datasets
import pandas as pd
import BaggingRobustRegressor

rng = np.random.RandomState(1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions:
def f(data_type='sin', n_samples=100, X=None):
        if data_type == 'sin':
                if X == None:
                        X = np.linspace(0, 6, n_samples)[:, np.newaxis]
                f = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
        elif data_type == 'exp':
                if X == None:
                        X = np.linspace(0, 6, n_samples)[:, np.newaxis]
                f = np.exp(-(X ** 2)) + 1.5 * np.exp(-((X - 2) ** 2)) + rng.normal(0, 0.1, X.shape[0])
        return X, f

def generate(data_type=None, n_samples=100, noise=0.1, n_repeat=1):
        if data_type == 'make_reg':
                X, y = sk.datasets.make_regression(n_samples=n_samples, n_features=50, n_informative=15, noise=noise, random_state=6)
        else:
                y = np.zeros((n_samples, n_repeat))
                for i in range(n_repeat):
                        X, y[:, i] = f(data_type, n_samples) + np.random.normal(0.0, noise, n_samples)
                X = X.reshape((n_samples, 1))
        return X, y

def get_dataset(data_type=None, test_size=0.2, n_samples=100, noise=0.1):
        if data_type == 'kc_house_data':
                dataset_link = '..//Datasets//kc_house_data.csv'
                dataset_df = pd.read_csv(dataset_link)
                dataset_df.drop("date", axis=1, inplace=True)
                # dataset_df = dataset_df / dataset_df.max()
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset_df, dataset_df['price'], test_size=test_size, random_state=3)
        elif data_type == 'carbon_nanotubes':
                dataset_link = '..//Datasets//carbon_nanotubes.csv'
                dataset_df = pd.read_csv(dataset_link, sep=';')
                for column in dataset_df.columns:
                        if dataset_df[column].dtype == 'object':
                                dataset_df[column] = [float(string.replace(',', '.')) for string in dataset_df[column]]
                X, y = dataset_df.iloc[:, :-3], dataset_df.iloc[:, -3:]
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=3)
        elif data_type == 'diabetes':
                dataset = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
                X, y = dataset[0], dataset[1]
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=3)

        else:
                X, y = generate(data_type=None, n_samples=n_samples, noise=noise, n_repeat=1)
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=3)
        n_samples = X_train.shape[0]
        return X_train, y_train, X_test, y_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
data_type = 'kc_house_data'  # kc_house_data / carbon_nanotubes / diabetes / sin / exp / make_reg
n_samples = 100
test_size = 0.2

train_noise = 0.1
n_estimators = 10
sigma0 = (750e3)**2  # noise variance

sigma_profile = sigma0 * np.ones([n_estimators, 1])
sigma_profile[1::2,:] = sigma0 / 100
noise_covariance = np.diag(sigma_profile.ravel())

X_train, y_train, X_test, y_test = get_dataset(data_type=data_type, test_size=test_size, n_samples=n_samples, noise=train_noise)

# Fit regression model
regr_1 = sklearn.ensemble.BaggingRegressor(sk.tree.DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, random_state=rng)
regr_2 = BaggingRobustRegressor.BaggingRobustRegressor(regr_1, noise_covariance, n_estimators, 'ben')
regr_3 = BaggingRobustRegressor.BaggingRobustRegressor(regr_1, noise_covariance, n_estimators, 'robust-bem')

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)
regr_3.fit(X_train, y_train)

# Predict: clean
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Plot the results
if X_test.shape[1] <= 1:
        plt.figure()
        plt.plot(X_test, y_test, c="k", label="Target", linestyle='-', linewidth=1)
        plt.scatter(X_train, y_train, c="k", label="Training")
        plt.plot(X_test, y_1, c="r", label="Clean", linestyle='--', linewidth=2)
        plt.plot(X_test, y_2, c="b", label="Prior", linestyle=':', linewidth=2)
        plt.plot(X_test, y_3, c="g", label="Robust", linestyle=':', linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Ensemble, T=" + str(n_estimators))
        plt.legend()
        plt.show(block=False)


plt.figure()
plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.5)
plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='.', markersize=0.5)
plt.plot(y_test, y_2, c="r", label="Prior", linestyle='', marker='o', markersize=0.5)
plt.plot(y_test, y_3, c="g", label="Robust", linestyle='', marker='*', markersize=0.5)
plt.grid()
plt.legend()

print("BEM MSE: \n\tNoiseless: " + str(sk.metrics.mean_squared_error(y_test, y_1, squared=False)) + \
                "\n\tPrior: " + str(sk.metrics.mean_squared_error(y_test, y_2, squared=False)) + \
                "\n\tRobust: " + str(sk.metrics.mean_squared_error(y_test, y_3, squared=False)) \
      )
