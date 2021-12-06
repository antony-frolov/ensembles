import numpy as np


def rmse(y_test, y_pred):
    return np.sqrt(((y_test - y_pred) ** 2).sum() / len(y_test))
