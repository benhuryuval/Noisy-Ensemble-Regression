import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.datasets
import pandas as pd


# Functions:
def f(data_type='sin', n_samples=100, X=None, rng=np.random.default_rng(seed=42)):
        if data_type == 'sin':
                if X == None:
                    X = np.linspace(0, 6, n_samples)[:, np.newaxis]
                f = np.sin(X).ravel() + np.sin(6 * X).ravel()# + rng.normal(0, 0.1, X.shape[0])
        elif data_type == 'exp':
                if X == None:
                    X = np.linspace(0, 6, n_samples)[:, np.newaxis]
                f = np.exp(-(X ** 2)).ravel() + 1.5 * np.exp(-((X - 2) ** 2)).ravel()# + rng.normal(0, 0.1, X.shape[0])
        if data_type == 'sin_outliers':
                if X == None:
                    X = np.linspace(0, 6, n_samples)[:, np.newaxis]
                f = np.sin(X).ravel() + np.sin(6 * X).ravel()
                n_outliers = round(n_samples/20)
                outlier_idxs, outlier_vals = rng.integers(n_samples, size=n_outliers), rng.integers(1, size=n_outliers)
                f[outlier_idxs] = outlier_vals * np.max(f) + (1-outlier_vals) * np.min(f)
                # f[outlier_idxs] *= 0
        return X, f

def generate(data_type=None, n_samples=100, noise=0.0, n_repeat=1, rng=np.random.default_rng(seed=42)):
        if data_type == 'make_reg':
                X, y = sk.datasets.make_regression(n_samples=n_samples, n_features=3, n_informative=2, noise=noise, random_state=0)
                y /= np.max(y)
        else:
                X, y = f(data_type, n_samples, rng=rng)
                y += rng.normal(0.0, noise, n_samples)
        return X, y

def get_dataset(data_type=None, n_samples=100, noise=0.1, rng=np.random.default_rng(seed=42)):
        # datasets_path = "..//Datasets//"
        datasets_path = "C://Users//Yuval//Documents//GitHub//Noisy-Ensemble-Regression//Datasets//"
        if data_type == 'kc_house_data':
                dataset_link = datasets_path + "kc_house_data.csv"
                dataset_df = pd.read_csv(dataset_link)
                dataset_df.drop("date", axis=1, inplace=True)
                X, y = dataset_df.drop(['price', 'id'], axis=1, inplace=False), dataset_df['price']
                X, y = X.head(1000), y.head(1000)
        elif data_type == 'white-wine':
                dataset_link = datasets_path + "winequality-white.csv"
                dataset_df = pd.read_csv(dataset_link, sep=';')
                X, y = dataset_df.drop('quality', axis=1, inplace=False), dataset_df['quality']
                X, y = X.head(1000), y.head(1000)
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
                X, y = generate(data_type, n_samples=n_samples, noise=noise, n_repeat=1, rng=rng)
                X, y = pd.DataFrame.from_records(X), pd.Series(y)
        # Standartization of dataset
        X, y = (X - X.mean()) / X.std(), (y - y.mean()) / y.std()
        # Randomize data order
        perm = rng.permutation(len(X))
        X, y = X.to_numpy()[perm], y.to_numpy()[perm]
        if (len(X.shape) == 1) or (X.shape[1] == 1):
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return X, y

def partition_dataset(data_type=None, test_size=0.2, n_samples=100, noise=0.1):
        X, y = get_dataset(data_type=data_type, n_samples=n_samples, noise=noise)
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,
                                                                               y,
                                                                               test_size=test_size,
                                                                               random_state=rng)
        return X_train, y_train, X_test, y_test

def is_psd_mat(matrix):
        return np.all(np.linalg.eigvals(matrix) >= 0)

def gradient_descent_scalar(gamma_init, grad_fun, cost_fun, max_iter=30000, min_iter=10, tol=1e-5, learn_rate=0.2, decay_rate=0.2):
        """ This function calculates optimal (scalar) argument with AdaGrad-style gradient descent method using an early stop criteria and
        selecting the minimal value reached throughout the iterations """

        # initializations
        cost_evolution = [np.array([[0.0]])] * max_iter
        gamma_evolution = [np.array([[0.0]])] * max_iter
        grad_evolution = [np.array([[0.0]])] * max_iter
        eps = 1e-15  # tolerance value for adagrad learning rate update
        step, i = np.array([[0.0]]), 0  # initialize gradient-descent step to 0, iteration index in evolution

        # perform remaining iterations of gradient-descent
        gamma_evolution[0] = gamma_init
        Gt = 0  # gradient accumulator for AdaGrad normalization
        for i in range(0, max_iter-1):
            # calculate grad and update cost function
            grad_evolution[i], cost_evolution[i] = grad_fun(gamma_evolution[i]), cost_fun(gamma_evolution[i])

            # check convergence
            if i > max(min_iter, 0) and np.abs(cost_evolution[i]-cost_evolution[i-1]) <= tol*np.abs(np.mean(cost_evolution[i-1:i+1])):
                break
            else:
                # update learning rate and advance according to AdaGrad
                # Gt = Gt + grad_evolution[i]**2  # np.sum(np.concatenate(grad_evolution[0:i+1]) ** 2)
                # learn_rate_upd = np.divide(learn_rate, np.sqrt(Gt + eps))
                # step = step.dot(decay_rate) - np.dot(learn_rate_upd, grad_evolution[i])
                # vanilla step
                step = -grad_evolution[i].dot(learn_rate)
                gamma_evolution[i+1] = gamma_evolution[i] + step
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Cost function visualization for debug
        if False:
            import matplotlib.pyplot as plt
            npts = 10000
            g_vec = np.linspace(-10, 10, npts)
            for g_idx, g_ in enumerate(g_vec):
                cost_vec = cost_fun(g_)

            fig_cost = plt.figure(figsize=(12, 8))
            plt.plot(g_vec, cost_vec, '.', label="Cost")
            plt.xlabel('gamma')
            plt.ylabel('Cost')
            plt.show(block=False)

            plt.plot(np.concatenate(gamma_evolution[0:i], axis=0)[:,0], np.array(cost_evolution[0:i]), 'o', label="GD")
            plt.close(fig_cost)
        # - - - - - - - - - - - - - - - - - - - - - - - - - -

        return cost_evolution, gamma_evolution, i


def gradient_descent(gamma_init, grad_fun, cost_fun, max_iter=30000, min_iter=10, tol=1e-5, learn_rate=0.2, decay_rate=0.2):
    """ This function calculates optimal (scalar) argument with AdaGrad-style gradient descent method using an early stop criteria and
    selecting the minimal value reached throughout the iterations """

    # initializations
    cost_evolution = [None] * max_iter
    gamma_evolution = [None] * max_iter
    grad_evolution = [None] * max_iter
    eps = 1e-15  # tolerance value for adagrad learning rate update
    vec_siz = len(gamma_init.ravel())
    step, i = np.array([np.zeros(vec_siz)]), 0  # initialize gradient-descent step to 0, iteration index in evolution

    # perform remaining iterations of gradient-descent
    gamma_evolution[0] = gamma_init
    Gt = 0  # gradient accumulator for AdaGrad normalization
    for i in range(0, max_iter - 1):
        # calculate grad and update cost function
        grad_evolution[i], cost_evolution[i] = grad_fun(gamma_evolution[i]), cost_fun(gamma_evolution[i])

        # check convergence
        if i > max(min_iter, 0) and np.abs(cost_evolution[i] - cost_evolution[i-1]) <= tol:
            break
        else:
            # update learning rate and advance according to AdaGrad
            # Gt = Gt + grad_evolution[i]**2  # np.sum(np.concatenate(grad_evolution[0:i+1]) ** 2)
            # learn_rate_upd = np.divide(learn_rate*np.ones(vec_siz), np.sqrt(Gt + eps))
            # learn_rate_upd = learn_rate * 1 / np.sqrt(Gt + eps)
            # step = decay_rate * step - np.dot(learn_rate_upd, grad_evolution[i])
            # step = decay_rate * step - learn_rate_upd * grad_evolution[i]
            # "vanilla" gd step
            step = - learn_rate * grad_evolution[i]
            gamma_evolution[i + 1] = gamma_evolution[i] + step
    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # LAE plotting for visualization and debug
    if False:
        import matplotlib.pyplot as plt
        fig_cost = plt.figure(figsize=(12, 8))
        plt.plot(range(0, i), cost_evolution[0:i], '.', label="Cost")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.show(block=False)
        plt.close(fig_cost)
    # - - - - - - - - - - - - - - - - - - - - - - - - - -

    return cost_evolution, gamma_evolution, i


def calc_error(x, y, criterion):
    if criterion == "mse":
        return np.sqrt(np.square(np.subtract(x.ravel(), y.ravel())).mean())
    elif criterion == "mae":
        return np.abs(np.subtract(x.ravel(), y.ravel())).mean()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
