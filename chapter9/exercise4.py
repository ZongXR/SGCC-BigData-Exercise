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
    print(a.max())
    print(a.min())
    # 2
    print(a.argmax(axis=0))
    # 3
    print(a.max(axis=1))
    print(a.min(axis=1))
    # 4
    print(a.max(axis=0))
    print(a.min(axis=0))