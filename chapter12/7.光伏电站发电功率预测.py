# -*- coding: utf-8 -*-
"""
光伏发电随机性大、可调用性能弱，大规模接入电网会致使电网运行不稳定。通过分析气象与发电功率之间的关系，可以预测光伏电站的发电功率，
有助于电力系统调度部门统筹安排常规发电和光伏发电的协调配合，提高电网运行的稳定。
本题数据集（光伏电站发电功率预测数据.xlsx）提供了两个光伏电站2019年6月-2019年8月的数据，样例数据见下表。
"""
from typing import Optional
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def feature_engineering(_df_: DataFrame, _encoder_: Optional[ColumnTransformer] = None) -> (DataFrame, ColumnTransformer):
    """
    特征工程\n
    :param _df_: 输入数据
    :param _encoder_: 编码器
    :return: 输出数据, 编码器
    """
    _result_ = _df_.copy()
    _result_["时间"] = pd.to_datetime(_result_["时间"])
    _result_["year"] = _result_["时间"].dt.year
    _result_["month"] = _result_["时间"].dt.month
    _result_["day"] = _result_["时间"].dt.day
    _result_["hour"] = _result_["时间"].dt.hour
    _result_ = _result_.drop(columns=["时间"])
    if _encoder_ is None:
        _encoder_ = ColumnTransformer([
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["光伏电站编号", "气象站编号"])
        ], remainder="passthrough", n_jobs=-1)
        return DataFrame(_encoder_.fit_transform(_result_), columns=_encoder_.get_feature_names_out()), _encoder_
    else:
        return DataFrame(_encoder_.transform(_result_), columns=_encoder_.get_feature_names_out()), _encoder_


if __name__ == '__main__':
    # (1) 请对光伏电站发电功率数据进行清洗，将缺失值用均值填充。
    data = pd.read_excel("./光伏电站发电功率预测数据.xlsx", sheet_name="训练集")
    data.loc[:, "功率"] = data[["光伏电站编号", "功率"]].groupby("光伏电站编号").transform(lambda x: x.fillna(x.mean()))
    print(data["功率"].isnull().sum())

    # (2) 利用随机森林建立预测功率的模型，探索各个特征对功率的贡献值（feature importance），选择合适的可视化技术展示。
    x_train, encoder = feature_engineering(data, None)
    y_train = x_train.pop("remainder__功率")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)
    result = dict(zip(model.feature_names_in_, model.feature_importances_))
    result = {k.split("__")[1]: v for k, v in result.items()}
    plt.figure(1)
    Series(result).plot(kind="bar")
    plt.show()

    # (3) 依据气象数据及光伏电站历史发电数据，进行光伏电站超短期（0-4h整点）功率预测。
    data_test = pd.read_excel("./光伏电站发电功率预测数据.xlsx", sheet_name="测试集")
    data_result = pd.read_excel("./光伏电站发电功率预测数据.xlsx", sheet_name="结果集")
    x_test, _ = feature_engineering(data_test, encoder)
    x_test.pop("remainder__功率")
    data_result, _ = feature_engineering(data_result, encoder)
    y_test = data_result.pop("remainder__功率")
    y_pred = model.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
