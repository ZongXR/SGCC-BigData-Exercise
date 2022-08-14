# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    n_dots = 200
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    y = np.sin(x) + 0.2 * np.random.rand(n_dots) - 0.1
    x = x.reshape(-1, 1)
    p = Pipeline([
        ("poly_features", PolynomialFeatures()),
        ("model", LinearRegression(n_jobs=-1))
    ])
    for degree in (2, 3):
        p.set_params(poly_features__degree=degree)
        p.fit(x, y)
        model: LinearRegression = p.named_steps["model"]
        print(p.score(x, y))
