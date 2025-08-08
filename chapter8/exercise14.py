# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2


if __name__ == '__main__':
    data = load_iris()
    x = data["data"]
    y = data["target"]
    sel = SelectKBest(score_func=chi2, k=2)
    sel.fit(x, y)
    print(np.array(data["feature_names"])[sel.get_support()])