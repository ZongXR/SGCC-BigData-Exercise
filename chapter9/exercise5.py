# -*- coding: utf-8 -*-
import numpy as np


if __name__ == '__main__':
    a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 25],
        [3, 4, 5, 6],
        [4, 5, 6, 11]
    ])
    # 1
    print(a.mean(), np.median(a))
    # 2
    print(a.var(axis=1))
    # 3
    print(a.std(axis=0))
    # 4
    print(a.cumsum(axis=1))