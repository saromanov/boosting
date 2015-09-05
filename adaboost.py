import numpy as np

class Adaboost:
    def __init__(self, init='normal'):
        '''
          Args:
              init - choose type of weights initialization (normal, partial)
        '''
        self.init = init
        self.hyp = []

    def _init_weights(self, m):
        W = np.ones((m,))
        if self.init == 'normal':
            W = np.random.normal(0,0.1, m)
        if self.init == 'partial':
            W = np.ones((m,))/m
        return W

    def _update_weights(self, W):
        return W/W.sum()

    def _error(self, x, y):
        return np.exp(-y * x)

    def addHypothesis(self, func):
        self.hyp.append(func)

    def fit(self, X, y):
        ''' Args:
            X - n^d array
            y - 1d array with same length as X
        '''
        n = X.shape[0]
        assert(n == y.shape[0])
        W = self._init_weights(X.shape[0])
        hypo = self.hyp[0]
        terr = 999999
        for i in range(len(self.hyp)):
            err = np.sum([W[j] * self._error(self.hyp[i](X[j]), y[i]) for j in range(n)])
            if err < terr:
                terr = err
                hypo = self.hyp[i]
            alpha = 0.5 * np.log((1 - err)/err)

            W = W * np.exp(alpha * y * self.hypo(X))
        return np.sign(np.sum([self.rate * hypo(x) for x in self.X]))

