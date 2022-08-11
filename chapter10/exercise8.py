# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./load.csv")
    result = data["load"].describe()[["max", "min", "mean"]]
    result["median"] = data["load"].median()
    print(result)
    # 2
    level = pd.cut(data["load"], [
        0,
        data["load"].mean() - data["load"].std(),
        data["load"].mean() + data["load"].std(),
        data["load"].mean() + 2 * data["load"].std(),
        data["load"].mean() + 3 * data["load"].std(),
        np.inf
    ], labels=["低", "正常", "高", "严重", "异常"])
    data["level"] = level.astype(str).apply(lambda xx: np.nan if xx == "nan" else xx)
    print(data)
    plt.figure()
    plt.hist(data["load"], density=True)
    x = np.arange(0, max(data["load"]))
    y = norm.pdf(x, loc=data["load"].mean(), scale=data["load"].std())
    plt.plot(x, y)
    plt.vlines([
        data["load"].mean() - data["load"].std(),
        data["load"].mean() + data["load"].std(),
        data["load"].mean() + 2 * data["load"].std(),
        data["load"].mean() + 3 * data["load"].std()
    ], ymin=0, ymax=0.006, linestyles=":")
    plt.show()
    # 3
    r3 = data["level"].value_counts()
    r3["严重"] = 0
    r3["异常"] = 0
    result3 = pd.DataFrame({
        "百分比": r3,
        "参考值": [0] * r3.shape[0]
    })
    result3 = result3.reindex(index=["低", "正常", "高", "严重", "异常"])
    result3.loc["低", "参考值"] = norm.cdf(data["load"].mean() - data["load"].std(), loc=data["load"].mean(), scale=data["load"].std()) - norm.cdf(0, loc=data["load"].mean(), scale=data["load"].std())
    result3.loc["正常", "参考值"] = norm.cdf(data["load"].mean() + data["load"].std(), loc=data["load"].mean(), scale=data["load"].std()) - norm.cdf(data["load"].mean() - data["load"].std(), loc=data["load"].mean(), scale=data["load"].std())
    result3.loc["高", "参考值"] = norm.cdf(data["load"].mean() + 2 * data["load"].std(), loc=data["load"].mean(), scale=data["load"].std()) - norm.cdf(data["load"].mean() + data["load"].std(), loc=data["load"].mean(), scale=data["load"].std())
    result3.loc["严重", "参考值"] = norm.cdf(data["load"].mean() + 3 * data["load"].std(), loc=data["load"].mean(), scale=data["load"].std()) - norm.cdf(data["load"].mean() + 2 * data["load"].std(), loc=data["load"].mean(), scale=data["load"].std())
    result3.loc["异常", "参考值"] = 1 - result3["参考值"].sum()
    result3["百分比"] = result3["百分比"] / result3["百分比"].sum()
    print(result3)