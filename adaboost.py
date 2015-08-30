import numpy as np

class Adaboost:
    def __init__(self, rate=0.001, init='standard'):
        self.rate = rate
        self.init = init
        self.hyp = []

    def _init_weights(self, m):
        W = np.ones((m,))
        if self.init == 'normal':
            W = np.random.normal(0,0.1, m)
        return W

    def _update_weights(self, W):
        return W/W.sum()

    def _update_rate(self, error):
        return 0.5 * np.log((1 - error)/error)

    def addHypothesis(self, func):
        self.hyp.append(func)

    def fit(self, X, y):
        n = X.shape[0]
        assert(n == y.shape[0])
        W = self._init_weights(X.shape[0])
        hypo = self.hyp[0]
        for i in range(n):
            for h in self.hyp:
                res = h(X[i])
        return np.sign(np.sum([self.rate * h(x) for x in self.X]))

