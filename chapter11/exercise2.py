# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
    model.fit(x, y)
    print(model.score(x, y))