# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import MultinomialNB


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    model = MultinomialNB(alpha=0.0001)
    model.fit(x, y)
    print(model.score(x, y))