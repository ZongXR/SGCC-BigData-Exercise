# -*- coding: utf-8 -*-
"""
随机森林在sklearn.datasets中可以自行加载load_breast_cancer数据，具体数据如下：
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
该题直接调用模型不需要对样本进行切分，参数设置为n_estimators=100，max_depth=5，其他参数无需设置，
请采用随机森林做分类，输出模型的准确率score(X, y)
"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data["data"]
    y = data["target"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
    model.fit(x, y)
    print(model.score(x, y))