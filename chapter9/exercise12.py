# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold


if __name__ == '__main__':
    data = load_iris()
    x = data["data"]
    y = data["target"]
    sel = VarianceThreshold(threshold=0.6)
    sel.fit(x, y)
    print(np.array(data["feature_names"])[sel.get_support()])
