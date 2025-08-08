# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import pearsonr


def get_pearsonr(x: pd.DataFrame) -> float:
    """
    计算相关系数\n
    :param x: 输入数据
    :return: 相关系数
    """
    return pearsonr(x["最大负荷"], x["最高温度"])[0]


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./weather_load.csv", index_col="日期")
    # 2
    data.index = pd.to_datetime(data.index)
    # 3
    data["季度"] = data.index
    data["季度"] = data["季度"].apply(lambda x: x.quarter)
    # 4
    quarter_pearsonr = data[["季度", "最大负荷", "最高温度"]].groupby("季度").apply(lambda x: get_pearsonr(x))
    quarter_pearsonr = quarter_pearsonr.sort_values(ascending=False)
    print(quarter_pearsonr)