# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr


def get_pearsonr(xx: np.ndarray, yy: np.ndarray):
    """
    计算相关系数\n
    :param xx: x
    :param yy: y
    :return: 相关系数
    """
    result = np.zeros(shape=(xx.shape[1], ))
    for i in range(0, xx.shape[1]):
        result[i] = pearsonr(xx[:, i], yy)[0]
    return result


if __name__ == '__main__':
    data = load_iris()
    x = data["data"]
    y = data["target"]
    sel = SelectKBest(score_func=get_pearsonr, k=3)
    sel.fit(x, y)
    print(np.array(data["feature_names"])[sel.get_support()])
