import numpy as np
from scipy.optimize import fmin

class LSBoost(boost.Boost):
    """ implementation of LS Boost algorithm
    https://arxiv.org/abs/1505.04243
    """
    def __init__(self):
        self.hyp = []
        super(LSBoost, self).__init__(self)

    def addHypothesis(self, func):
        self.hyp.append(func)

    def _find_mult(self, p):
        return self._loss(self.hypprev - p * self.gr, self.y)

    def fit(self, X, y):
        assert(len(self.hyp) > 0)
        hypprev = self.hyp[0](X)
        n = X.shape[0]
        self.y = y
        for hyp in range(1, self.hyp):
            ydot = [y[i] - hypprev(X[i]) for i in range(n)]
            loss1 = hypprev - self.grad(ydot)
            self.hypprev = hypprev
            p = fmin(self._find_mult)
            hypprev = hypprev + p * self.grad(a)

