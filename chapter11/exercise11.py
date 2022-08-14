# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    x = [2, 3, 6, 9, 11, 7]
    y = [3.9, 6, 11.9, 18, 22, 14]
    model = LinearRegression(n_jobs=-1)
    model.fit(np.array(x).reshape(-1, 1), y)
    print(model.score(np.array(x).reshape(-1, 1), y))