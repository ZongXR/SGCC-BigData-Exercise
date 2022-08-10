# -*- coding: utf-8 -*-
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel


if __name__ == '__main__':
    data = load_boston()
    x = data["data"]
    y = data["target"]
    sel = SelectFromModel(Lasso(alpha=0.5))
    sel.fit(x, y)
    print(data["feature_names"][sel.get_support(False)])
