import numpy as np
import boost

class Adaboost(boost.Boost):
    def __init__(self, init='partial'):
        '''
          Args:
              init - choose type of weights initialization (normal, partial)
        '''
        super(Adaboost, self).__init__(self)
        self.init = init
        self.hyp = []
        self.rates = []
        self.hypo = None

    def _init_weights(self, m):
        W = np.ones((m,))
        if self.init == 'normal':
            W = np.random.normal(0,0.5, m)
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
        n = self._get_dataset_size(X)
        W = self._init_weights(n)
        assert(len(self.hyp) > 0)
        hypo = self.hyp[0]
        self.rate = 0.001
        if self._is_numpy(X) is False:
            X = self._to_numpy(X)
        if self._is_numpy(y) is False:
            y = self._to_numpy(y)
        for i in range(len(self.hyp)):
            err = np.sum([W[j] * self._loss(self.hyp[i](X[j]), y[i]) for j in range(n)])
            alpha = 0.5 * np.log((1 - err)/err)
            self.rates.append(alpha)
            W = W * np.exp(-alpha)
        self.hypo = hypo

    def _output(self, X, hypos):
        for x in X:
            print(np.sign(np.sum([self.rates[i] * hypos[i](x) for i in range(len(hypos))])))

    def predict(self, X):
        return self._output(X, self.hyp)
