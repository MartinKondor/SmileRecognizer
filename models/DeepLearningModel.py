import numpy as np


def act_f(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# not a really ordinary nn but it will work
class DeepLearningModel(object):
    
    def __init__(self, biggest_img_size: tuple=(325, 325,)):
        self.learning_rate = .5
        self.biggest_img_size = biggest_img_size

        self.hls = biggest_img_size[0]
        self.ils = biggest_img_size[0]
        self.ols = 1

        self.Wxh = 2*np.random.random((self.ils, self.hls)) - 1
        self.Why = 2*np.random.random((self.hls, self.ols)) - 1

    def train(self, _X: np.ndarray, _t: np.ndarray):
        X = self.preprocessing(_X)

        h = act_f(np.dot(X, self.Wxh))
        o = np.array([act_f(np.dot(h, self.Why).sum())])
        
        dh = None
        do = None

        exit()

    def preprocessing(self, _X):
        c = np.zeros(self.biggest_img_size)
        for i in range(_X.shape[0]):
            for j in range(_X.shape[1]):
                c[i][j] += _X[i][j]
        return c

    def test(self, X: np.ndarray):
        h = act_f(np.dot(X, self.Wxh))
        return np.array([act_f(np.dot(h, self.Why).sum())])
