import numpy as np

class Boost:
    def __init__(self, lrate=0.001):
        self.lrate = lrate

    def _square_loss(self, x, y):
        return (x - y)**2

    def _logistic_loss(self, x, y):
        return x * np.log(1 + np.exp(-y)) + (1 - x) * np.log(1 + np.exp(y))

    def _find_mult(self, p):
        return self._loss(self.hypprev - p * self.gr, self.y)

    def _get_dataset_size(self, item):
        return item.shape[0] if type(item).__module__ == np.__name__ else len(item)

