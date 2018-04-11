import numpy as np
from matplotlib import pyplot as plt
from random import random

from sklearn.model_selection import train_test_split


def get_error(y, new_y):
    return 1.0/(2*y.shape[0]) * np.sum((y - new_y) ** 2)

def get_model(x, y):
    decay = 0.05
    print(x.size)
    w = np.random.rand(x.shape[1], 1)
    for _ in range(10000):
        new_y = np.dot(x, w)
        error = get_error(y, new_y)
        error_der = np.sum(y - new_y)
        w = w - decay * error_der * x

        print(error)



def get_samples(n_dim=5, n_samples=10000):
    x = np.random.rand(n_samples, n_dim) * 1000 - 500
    w = np.linspace(1, n_dim + 1, n_dim).reshape(n_dim, 1)
    y = np.dot(x, w) + np.random.randn(n_samples, 1) * 100 - 50

    return x, y

def main():
    x, y = get_samples()
    print(x.size)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    f = get_model(x_train, y_train)


if __name__ == '__main__':
    main()
