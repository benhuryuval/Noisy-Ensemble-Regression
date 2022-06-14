import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.datasets
import pandas as pd

rng = np.random.RandomState(42)

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
        datasets_path = "..//..//Datasets//"
        if data_type == 'kc_house_data':
                dataset_link = datasets_path + "kc_house_data.csv"
                dataset_df = pd.read_csv(dataset_link)
                dataset_df.drop("date", axis=1, inplace=True)
                X, y = dataset_df.drop('price', axis=1, inplace=False), dataset_df['price']
        elif data_type == 'white-wine':
                dataset_link = datasets_path + "winequality-white.csv"
                dataset_df = pd.read_csv(dataset_link, sep=';')
                X, y = dataset_df.drop('quality', axis=1, inplace=False), dataset_df['quality']
        elif data_type == 'diabetes':
                dataset = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
                X, y = dataset[0], dataset[1]
        elif data_type == 'carbon_nanotubes':
                dataset_link = datasets_path + "carbon_nanotubes.csv"
                dataset_df = pd.read_csv(dataset_link, sep=';')
                for column in dataset_df.columns:
                        if dataset_df[column].dtype == 'object':
                                dataset_df[column] = [float(string.replace(',', '.')) for string in dataset_df[column]]
                X, y = dataset_df.iloc[:, :-3], dataset_df.iloc[:, -3:]
        elif data_type == 'auto-mpg':
                dataset_link = datasets_path + "auto-mpg.csv"
                dataset_df = pd.read_csv(dataset_link)
                X, y = dataset_df['weight'], dataset_df['mpg']
        else:
                X, y = generate(data_type, n_samples=n_samples, noise=noise, n_repeat=1)
                X, y = pd.DataFrame.from_records(X), pd.Series(y)
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,
                                                                               y,
                                                                               test_size=test_size,
                                                                               random_state=42)
        return X_train, y_train, X_test, y_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
