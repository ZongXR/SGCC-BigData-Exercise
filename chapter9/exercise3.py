# -*- coding: utf-8 -*-
from scipy.stats import norm


if __name__ == '__main__':
    # 1
    result = norm.pdf(x=range(1, 12), loc=0, scale=1)
    print(result)
    # 2
    print(result[5])