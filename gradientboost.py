''' Implementation of gradient boosting
'''
import numpy as np
from scipy.optimize import line_search


class GradientBoost:
    def __init__(self, lrate=0.001):
        ''' In the case if learning rate is const
        '''
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

    def _negative_gradient(self, loss, X, y):
        pass

    def fit(self, X, y, iters=100):
        n = X.shape[0]
        assert(n == y.shape[0])
        assert(len(hyp) > 0)
        params = np.ones(n)
        prev = self.hyp[0](X)
        for i in range(iters):
            grad = self._negative_gradient(self._loss, X, y)
            prev = prev + self.lrate * self.hyp[i](X)
        return prev



