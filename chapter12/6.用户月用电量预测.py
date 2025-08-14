# -*- coding: utf-8 -*-
"""
用户用电量反映了企业发展情况和居民生活水平，通过用户历史月电量数据和用户基本信息，可以从不同维度分析客户的用电行为并进行电量预测，为公司的精细化管理提供支撑。
本题数据集提供了50个用户2014年1月-2019年7月的数据，数据表（用户月用电量预测数据.xlsx）结构见下表。
"""
from typing import Optional
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def get_season(_month_) -> str:
    """
    计算季节\n
    :param _month_: 年月
    :return: 季节
    """
    _month_ = _month_ % 100
    if _month_ in (1, 2, 3):
        return "spring"
    elif _month_ in (4, 5, 6):
        return "summer"
    elif _month_ in (7, 8, 9):
        return "autumn"
    elif _month_ in (10, 11, 12):
        return "winter"
    else:
        raise ValueError("月份不对")


def feature_engineering(_df_: DataFrame, encoder: Optional[ColumnTransformer] = None) -> (DataFrame, Pipeline):
    """
    特征工程\n
    :param _df_: 输入数据
    :param encoder: 编码器
    :return: 输出数据, 编码器
    """
    _result_ = _df_.copy()
    _result_["year"] = _result_["时间"].apply(lambda x: x // 100)
    _result_["month"] = _result_["时间"].apply(lambda x: x % 100)
    _result_["season"] = _result_["时间"].apply(get_season)
    _result_ = _result_.drop(columns=["时间"])
    if encoder is None:
        onehot = ColumnTransformer(transformers=[
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["户号", "类型", "season"])
        ], remainder="passthrough", n_jobs=-1)
        encoder = Pipeline([
            ("onehot", onehot),
            ("standard", StandardScaler())
        ])
        return DataFrame(encoder.fit_transform(_result_), columns=encoder.get_feature_names_out()), encoder
    else:
        return DataFrame(encoder.transform(_result_), columns=encoder.get_feature_names_out()), encoder


if __name__ == '__main__':
    # (1) 统计每个用户的总电量、平均电量、最大电量、最小电量，最终结果按照下表参考文件结构保存为统计用户用电量情况表.csv
    data_train = pd.read_excel("./用户月用电量预测数据.xlsx", sheet_name="训练集")
    data_test = pd.read_excel("./用户月用电量预测数据.xlsx", sheet_name="测试集")
    data_result = pd.read_excel("./用户月用电量预测数据.xlsx", sheet_name="结果集")
    result = data_result[["户号", "电量"]].groupby("户号").agg(["sum", "mean", "max", "min"])
    result.columns = result.columns.droplevel(0)
    result = result.rename(columns={
        "sum": "总电量",
        "mean": "平均电量",
        "max": "最大电量",
        "min": "最小电量"
    })
    result.to_csv("./用户用电量情况表.csv", index=True, index_label="用户")

    # (2) 请根据给出的月电量，简要分析在不同季节、不同容量下的用电量特性（可视化分析）。
    data_train["season"] = data_train["时间"].apply(get_season)
    plt.figure(1)
    data_train[["season", "电量"]].groupby("season").sum()["电量"].reindex(index=[
        "spring", "summer", "autumn", "winter"
    ]).plot()
    plt.show()
    plt.figure(2)
    data_train[["运行容量", "电量"]].groupby("运行容量").sum()["电量"].sort_index(ascending=True).plot()
    plt.show()

    # (3) 基于时间序列技术，预测每个用户下一个月的电量，采用MSE作为评价指标。
    x_train, trans = feature_engineering(data_train, None)
    x_test, _ = feature_engineering(data_test, trans)
    data_result, _ = feature_engineering(data_result, trans)
    y_train = x_train.pop("remainder__电量")
    x_test.pop("remainder__电量")
    y_test = data_result["remainder__电量"]
    model = SVR()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
