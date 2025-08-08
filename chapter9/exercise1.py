# -*- coding: utf-8 -*-
from scipy.stats import binom


if __name__ == '__main__':
    # 1 二项分布的概率质量分布
    probs = binom.pmf(k=range(6), n=5, p=0.8)
    print(probs)
    # 2 发生0次的概率
    print(probs[0])

