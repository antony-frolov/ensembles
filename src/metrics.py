def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).sum() / len(y_test)
