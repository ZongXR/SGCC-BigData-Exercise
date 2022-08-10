# -*- coding: utf-8 -*-
from scipy.stats import poisson


if __name__ == '__main__':
    # 泊松分布
    print(poisson.pmf(k=6, mu=1))