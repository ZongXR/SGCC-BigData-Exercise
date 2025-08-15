# -*- coding: utf-8 -*-
"""
在sklearn.datasets中可以自行加载load_breast_cancer数据，具体数据如下：
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
该题直接调用模型不需要对样本进行切分，参数设置为max_depth=i，i为range(5, 10)，请采用决策树做分类输出不同max_depth下的model.score(X, y)
"""
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