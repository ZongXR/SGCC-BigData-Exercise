# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    num1 = [3, 60, 50, 43, 70, 52, 26, 37, 0, 80, 56, 77, 35, 67, 100]
    # 1
    r1, b1 = pd.cut(num1, bins=[0, 30, 60, 90, 100], labels=["组1", "组2", "组3", "组4"], include_lowest=True, retbins=True)
    print(r1.tolist())
    print(b1)
    # 2
    r2, b2 = pd.qcut(num1, 5, labels=["组1", "组2", "组3", "组4", "组5"], retbins=True)
    print(r2.tolist())
    print(b2)
