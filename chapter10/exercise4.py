# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    model = SVC(kernel="rbf", gamma=0.1)
    model.fit(x, y)
    print(model.score(x, y))