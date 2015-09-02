''' Implementation of gradient boosting
'''
import numpy as np
from scipy.optimize import line_search


class GradientBoost:
    def __init__(self, lrate=0.001):
        self.lrate = lrate
        self.hyp = []

    def addHypothesis(self, func):
        self.hyp.append(func)

    def _update(self, items):
        pass

    def _square_loss(self, x, y):
        return (x - y)**2

    def _logistic_loss(self, x, y):
        return x * np.log(1 + np.exp(-y)) + (1 - x) * np.log(1 + np.exp(y))

    def fit(self, X, y, iters=100):
        n = X.shape[0]
        assert(n == y.shape[0])
        params = np.ones(n)
        for i in range(iters):
            grad = self._loss(X, y)

