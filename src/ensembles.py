from timeit import default_timer

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor

from metrics import rmse


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth

        if feature_subsample_size is None:
            feature_subsample_size = 1 / 3
        self.feature_subsample_size = feature_subsample_size

        self.estimator = lambda: DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)

        self.estimators = None
        self.feature_subsamples = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        self.estimators = []
        self.feature_subsamples = []
        hist = {'n_estimators': [], 'time': [], 'train_rmse': []}
        if X_val is not None and y_val is not None:
            hist['val_rmse'] = []

        features = np.arange((X.shape[1]))
        feature_subsample_size = int(np.floor(X.shape[1] * self.feature_subsample_size))

        start_time = default_timer()
        for i in range(self.n_estimators):
            estimator = self.estimator()

            feature_subsample = np.random.permutation(features)[:feature_subsample_size]
            self.feature_subsamples.append(feature_subsample)

            objs = np.arange((X.shape[0]))
            obj_subsample = np.random.choice(objs, size=X.shape[0])
            obj_idxs, feat_idxs = np.meshgrid(obj_subsample, feature_subsample, indexing='ij')
            X_subsample = X[obj_idxs, feat_idxs]
            y_subsample = y[obj_subsample]
            estimator.fit(X_subsample, y_subsample)

            self.estimators.append(estimator)

            y_pred = self.predict(X)
            hist['train_rmse'].append(rmse(y, y_pred))
            hist['n_estimators'].append(i+1)
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                hist['val_rmse'].append(rmse(y_val, y_pred))

            hist['time'].append(default_timer() - start_time)

            print(f'\rEstimator {i+1}/{self.n_estimators} trained', end='')
        print()

        return hist

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = []
        for i, estimator in enumerate(self.estimators):
            y_pred = estimator.predict(X[:, self.feature_subsamples[i]])
            predictions.append(y_pred)

        return np.mean(predictions, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        if feature_subsample_size is None:
            feature_subsample_size = 1 / 3
        self.feature_subsample_size = feature_subsample_size

        self.estimator = lambda: DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)

        self.estimators = None
        self.feature_subsamples = None
        self.weights = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        self.estimators = []
        self.feature_subsamples = []
        self.weights = []

        f = np.zeros((X.shape[0],))
        hist = {'n_estimators': [], 'time': [], 'train_rmse': []}
        if X_val is not None and y_val is not None:
            hist['val_rmse'] = []

        features = np.arange((X.shape[1]))
        feature_subsample_size = int(np.floor(X.shape[1] * self.feature_subsample_size))

        start_time = default_timer()
        for i in range(self.n_estimators):
            estimator = self.estimator()

            feature_subsample = np.random.permutation(features)[:feature_subsample_size]
            self.feature_subsamples.append(feature_subsample)

            gradient = 2 * (f - y)
            estimator.fit(X[:, feature_subsample], -gradient)
            predictions = estimator.predict(X[:, feature_subsample])
            a = minimize_scalar(lambda a: np.sum((f + a * predictions - y) ** 2)).x
            self.weights.append(a)
            f = f + self.learning_rate * a * predictions

            self.estimators.append(estimator)

            y_pred = self.predict(X)
            hist['train_rmse'].append(rmse(y, y_pred))
            hist['n_estimators'].append(i+1)
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                hist['val_rmse'].append(rmse(y_val, y_pred))

            hist['time'].append(default_timer() - start_time)

            print(f'\rEstimator {i+1}/{self.n_estimators} trained', end='')
        print()

        return hist

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        y_pred = np.zeros((X.shape[0],))
        for i, estimator in enumerate(self.estimators):
            prediction = estimator.predict(X[:, self.feature_subsamples[i]])
            y_pred += self.learning_rate * self.weights[i] * prediction

        return y_pred
