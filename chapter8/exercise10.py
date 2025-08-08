# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("./9.3/example.csv", encoding="gbk")
    result = pd.get_dummies(data, columns=["线路名称"])
    print(result)