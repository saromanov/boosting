import numpy as np

class Boost:
    """ Basic class for boosting algorithms
    this class contains helpful methods
    like loss, logistic regression methods and
    basic methods like fit and predict
    """ 
    def __init__(self, lrate=0.001):
        self.lrate = lrate

    def _loss(self, x, y):
        return x != y

    def _square_loss(self, x, y):
        return (x - y)**2

    def _logistic_loss(self, x, y):
        return x * np.log(1 + np.exp(-y)) + (1 - x) * np.log(1 + np.exp(y))

    def _find_mult(self, p):
        return self._loss(self.hypprev - p * self.gr, self.y)

    def _get_dataset_size(self, item):
        return item.shape[0] if type(item).__module__ == np.__name__ else len(item)

    def _is_numpy(self, item):
        return True if type(item).__module__ == np.__name__ else False

    def _to_numpy(self, item):
        return np.array(item)

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

