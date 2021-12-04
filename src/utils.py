from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np


class DataPreprocessor():
    def __init__(self, mode='auto', num_features=None, bin_features=None, cat_features=None):
        if mode not in ['auto', 'manual']:
            raise ValueError()
        self.mode = mode

        self.transformer = None
        if mode == 'manual':
            self.transformer = ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_features),
                ('std_scaling', StandardScaler(), num_features),
                ('passthrough', 'passthrough', bin_features)
                ])

    def fit(self, X):
        if self.mode == 'auto':
            bin_features = X.columns[(X.dtypes == int) &
                                     np.array([X[col].unique().size == 2 for col in X.columns])].to_list()
            cat_features = X.columns[X.dtypes == np.dtype('O')].to_list()
            num_features = X.columns[~X.columns.isin(bin_features + cat_features)]

            self.transformer = ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_features),
                ('std_scaling', StandardScaler(), num_features),
                ('passthrough', 'passthrough', bin_features)
                ])

        self.transformer.fit(X)

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
