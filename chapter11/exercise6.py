# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    model = LogisticRegression(solver="liblinear")
    model.fit(x, y)
    print(model.score(x, y))