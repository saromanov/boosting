''' Implementation of gradient boosting
'''
import numpy as np
import boost
from scipy.optimize import line_search


class GradientBoost(boost.Boost):
    def __init__(self, lrate=0.001):
        """ In the case if learning rate is const
        """
        super(GradientBoost, self).__init__(self)
        self.lrate = lrate
        self.dfunc = None
        self.hyp = []

    def add_hypothesis(self, func):
        self.hyp.append(func)

    def add_diff_Func(self, func):
        """ add differential loss function
        """
        self.dfunc = func

    def _update(self, items):
        pass

    def _square_loss(self, x, y):
        return (x - y)**2

    def _logistic_loss(self, x, y):
        return x * np.log(1 + np.exp(-y)) + (1 - x) * np.log(1 + np.exp(y))

    def _negative_gradient(self, loss, X, y):
        return 1-y

    def _get_dataset_size(self, item):
        return item.shape[0] if type(item).__module__ == np.__name__ else len(item)

    def find_rate(self, phi, *args):
        x = args[0]
        prev = args[1]
        hm = args[2]
        y = args[3]
        return np.sum(self._logistic_loss(prev(x) + phi * hm(x), y))


    def fit(self, X, y, iters=100):
        """ fitting of the model
        
            :param X: n^d array
            :param y 1d array with same length as X
        """
        n = self._get_dataset_size(X)
        ny = self._get_dataset_size(y)
        assert(len(self.hyp) > 0)
        params = np.ones(n)
        prev = [self.hyp[0](x) for x in X]
        h = self.hyp[0](X)
        smallerr = np.sum(self._logistic_loss(h(X), y))
        for i in range(1, len(self.hyp)):
            current = np.sum(self._logistic_loss(self.hyp[0](X), y))
            grads = [self.dfunc(self._loss, X[j], y[j]) for j in range(n)]
            if current < smallerr:
                smallerr = current
                h = self.hyp[i]
            lrate = minimize(self.find_rate, args=(X, prev, h, y), method='L-BFGS-B')
            prev = prev + lrate * [self.hyp[i](x) for x in X]
        return prev


    def predict(self, X):
        for h in self.hyp:
            result = h(X)



