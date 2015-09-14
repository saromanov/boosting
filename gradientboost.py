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
        ''' Args:
              X - dataset of training points
              y - dataset of training labels
        '''
        n = X.shape[0]
        assert(n == y.shape[0])
        assert(len(hyp) > 0)
        params = np.ones(n)
        prev = self.hyp[0](X)
        for i in range(iters):
            grads = [self._negative_gradient(self._loss, X[i], y[i]) for i in range(n)]
            h = self.hyp[0](X)
            smallerr = np.sum(self._logistic_loss(h(X), y))
            for i in range(1, len(self.hyp)):
                current = np.sum(self._logistic_loss(self.hyp[0](X), y))
                if current < smallerr:
                    smallerr = current
                    h = self.hyp[i]
            self.lrate = np.sum(self._loss(prev + self.lrate * h(X)))
            prev = prev + self.lrate * self.hyp[i](X)
        return prev

    def predict(self, X):
        for h in self.hyp:
            pass



