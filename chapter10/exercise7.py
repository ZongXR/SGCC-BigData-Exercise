# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import poisson


if __name__ == '__main__':
    # 1 泊松分布的概率质量分布pmf
    print(poisson.pmf(k=0, mu=2))
    # 2 泊松分布的累计分布函数cdf
    print(1 - sum(poisson.pmf(k=range(0, 3), mu=3)))
    print(1 - poisson.cdf(k=2, mu=3))
    # 3
    for mu in np.arange(0, 10, 0.1):
        prob = 1 - poisson.cdf(k=2, mu=mu)
        if prob > 0.8:
            print(mu)
            break
