import numpy as np

class LSBoost:
    def __init__(self):
        self.hyp = []

    def addHypothesis(self, func):
        self.hyp.append(func)

    def fit(self, X, y):
        hypprev = self.hyp[0](X)
        n = X.shape[0]
        for hyp in range(1, self.hyp):
            ydot = [y[i] - hypprev(X[i]) for i in range(n)]
            hypprev = hypprev + self.grad(a)

