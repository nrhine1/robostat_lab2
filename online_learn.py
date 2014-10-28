import numpy as np
import os,sys
import sklearn.metrics


class online_learner(object):
    def __init__(self):
        pass

    def predict(x):
        pass

    def fit(x, y):
        pass

    def evaluate(X,Y):
        n_samples = X.shape[0]
        Y_hat = np.zeros(Y.shape)
        for (xi, x) in enumerate(X):
            y_hat = predict(x)
            Y_hat[xi] = y_hat
        # Compute confusion mat
        confusion_mat = sklearn.metrics.confusion_matrix(Y, Y_hat, labels=None)


class online_logistic(online_learner):
    def __init__(self, l2_lam):
        pass

    def predict(x):
        pass

    def fit(x, y):
        pass


