# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.ensemble
import sklearn.datasets
import pandas as pd
import BaggingRobustRegressor

rng = np.random.RandomState(42)

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
                f = np.exp(-(X ** 2)).ravel() + 1.5 * np.exp(-((X - 2) ** 2)).ravel() + rng.normal(0, 0.1, X.shape[0])
        return X, f

def generate(data_type=None, n_samples=100, noise=0.1, n_repeat=1):
        if data_type == 'make_reg':
                X, y = sk.datasets.make_regression(n_samples=n_samples, n_features=15, n_informative=10, noise=noise, random_state=42)
        else:
                X, y = f(data_type, n_samples)
                y += np.random.normal(0.0, noise, n_samples)
        return X, y

def get_dataset(data_type=None, test_size=0.2, n_samples=100, noise=0.1):
        if data_type == 'kc_house_data':
                dataset_link = '..//Datasets//kc_house_data.csv'
                dataset_df = pd.read_csv(dataset_link)
                dataset_df.drop("date", axis=1, inplace=True)
                # dataset_df = dataset_df / dataset_df.max()
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset_df, dataset_df['price'], test_size=test_size, random_state=42)
        elif data_type == 'carbon_nanotubes':
                dataset_link = '..//Datasets//carbon_nanotubes.csv'
                dataset_df = pd.read_csv(dataset_link, sep=';')
                for column in dataset_df.columns:
                        if dataset_df[column].dtype == 'object':
                                dataset_df[column] = [float(string.replace(',', '.')) for string in dataset_df[column]]
                X, y = dataset_df.iloc[:, :-3], dataset_df.iloc[:, -3:]
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
        elif data_type == 'diabetes':
                dataset = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
                X, y = dataset[0], dataset[1]
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
        else:
                X, y = generate(data_type, n_samples=n_samples, noise=noise, n_repeat=1)
                X, y = pd.DataFrame.from_records(X), pd.Series(y)
                X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, y_train, X_test, y_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
data_type = 'make_reg'  # kc_house_data / diabetes / sin / exp / make_reg
# / carbon_nanotubes (cant use, 3dim target y)
n_samples = 1000
test_size = 0.5
train_noise = 0.1

n_estimators = 10
tree_max_depth = 6
snr_db = -20

snr = 10**(snr_db/10)

# Get data set
X_train, y_train, X_test, y_test = get_dataset(data_type=data_type, test_size=test_size, n_samples=n_samples, noise=train_noise)
X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

# Set prediction noise
sig_var = np.var(y_train)
sigma0 = sig_var/snr  # noise variance
# sigma0 = (200e3)**2  # noise variance

sigma_profile = sigma0 * np.ones([n_estimators, 1])
sigma_profile[1::2,:] = sigma0 / 100
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

# Plot model accuracy (no noise)
plt.figure()
plt.plot(y_test, y_test, c="k", label="Target", linestyle='-', marker='', linewidth=0.25)
plt.plot(y_test, y_1, c="k", label="Noiseless", linestyle='', marker='o', markersize=1)

# Plot the results
if X_test.shape[1] <= 1:
        ind = np.argsort(X_test[:,0])
        plt.figure()
        plt.plot(X_test[ind], y_test[ind], c="k", label="Target", linestyle='-', linewidth=1)
        plt.scatter(X_train, y_train, c="k", label="Training", marker='o')
        plt.plot(X_test[ind], y_1[ind], c="r", label="Clean", linestyle='--', linewidth=2)
        plt.plot(X_test[ind], y_2[ind], c="b", label="Prior", linestyle=':', linewidth=2)
        plt.plot(X_test[ind], y_3[ind], c="g", label="Robust", linestyle=':', linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
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
