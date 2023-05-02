import numpy as np
np.random.seed(211)


class cross_entropy():
    def __init__(self):
        pass
    def forward(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + 1e-8))
    def gradient(self, y_pred, y):
        return -y / (y_pred + 1e-8)