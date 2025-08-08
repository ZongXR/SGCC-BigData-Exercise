# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVR


if __name__ == '__main__':
    n_dots = 200
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    y = np.sin(x) + 0.2 * np.random.rand(n_dots) - 0.1
    model = SVR()
    model.fit(x.reshape(-1, 1), y)
    print(model.score(x.reshape(-1, 1), y))

