# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans


def get_features(_df: pd.DataFrame) -> pd.Series:
    """
    计算特征\n
    :param _df: 输入数据
    :return: 返回一行
    """
    _result = pd.Series({
        "load_max": _df["负载率"].max(),
        "load_min": _df["负载率"].min(),
        "load_mean": _df["负载率"].mean(),
        "load_std": _df["负载率"].std()
    })
    return _result


if __name__ == '__main__':
    data = pd.read_csv("./聚类1-基于划分聚类算法的变压器状态分析检测/transformer_overload.csv")
    data["时间"] = pd.to_datetime(data["时间"])
    data["负载率"] = data["电压(KV)"] * data["电流(A)"] / data["额定容量(KVA)"]
    x = data.groupby("变压器").apply(lambda xx: get_features(xx))
    model = KMeans(n_clusters=4, n_jobs=-1)
    model.fit(x)
    result = x.copy()
    result["label"] = model.labels_
    print(result)