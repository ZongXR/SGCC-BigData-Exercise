# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    for max_depth in range(5, 10):
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(x, y)
        print(model.score(x, y))