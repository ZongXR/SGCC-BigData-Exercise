# -*- coding: utf-8 -*-
"""
油温是变压器运行所关注的重要指标，对油温的监测有助于跟踪变压器的健康状况。影响油温的因素包括负荷、气温等。
数据集情况：变压器油温和负荷数据主要取自地调D5000系统，气象数据来源于电网气象灾害监测预警系统，具体数据来源系统、数据字段、数据取值等情况见下表。
"""
from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def pre_processing(df: DataFrame) -> DataFrame:
    """
    数据预处理\n
    :param df: 输入数据
    :return: 输出数据
    """
    result = df.copy()
    result["时间"] = pd.to_datetime(result["时间"])
    result = result.rename(columns={
        "Unnamed: 1": "年份"
    })
    return result


def feature_engineering(df: DataFrame, encoder: Optional[ColumnTransformer] = None) -> (DataFrame, OneHotEncoder):
    """
    特征工程\n
    :param df: 输入数据
    :param encoder: 编码器
    :return: 输出数据
    """
    df = df.dropna()
    df = df.drop(columns=["时间", "年份", "地区", "变电站电压等级", "主变名称", "母线名称"])
    if encoder is None:
        encoder = ColumnTransformer(transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["变电站名称"])
        ], remainder="passthrough", n_jobs=-1)
        return DataFrame(encoder.fit_transform(df), columns=encoder.get_feature_names_out()), encoder
    else:
        return DataFrame(encoder.transform(df), columns=encoder.get_feature_names_out()), encoder


if __name__ == '__main__':
    # (1) 请根据给出的油温数据（变压器油温预测数据集_训练集.csv）在一张图内绘制不同变压器的油温数据折线图，横坐标标签为2018-01、2018-03、2018-05、……、2019-01，
    # 不同变压器线条以不同颜色区分，并在图例的左上角生成对应的标签，图标题为oil_temperature，最后保存为oil_temperature.png。
    oil_temperature = pd.read_csv("./变压器油温预测数据集_训练集.csv", encoding="gbk")
    oil_temperature = pre_processing(oil_temperature)
    plt.figure(1)
    for trans_name in oil_temperature["变电站名称"].unique():
        trans = oil_temperature[oil_temperature["变电站名称"] == trans_name]
        plt.plot(trans["时间"], trans["油温(℃)"], label=trans_name)
    plt.title("oil_temperature")
    plt.legend(loc='upper left')
    plt.savefig("./oil_temperature.png")
    plt.show()

    # (2) 请根据给出的油温和负荷数据（变压器油温预测数据集_训练集.csv），利用Bootstrap每次采90%的样本，
    # 分析采样数据的负荷、气温、风速、风向等因素与油温之间的pearson相关系数r，重复100次，估计相关系数的95%置信区间（2.5%-97.5%）。
    # 最终结果按照下表的参考文件结构保存为变压器油温分析.csv。
    results = []
    for _ in tqdm(range(100)):
        df = oil_temperature.sample(frac=0.9, random_state=42)[["油温(℃)", "电流值(A)", "温度(℃)", "10分钟最大风速(m/s)", "最大风速时风向"]].dropna()
        corr = df.corr()["油温(℃)"][["电流值(A)", "温度(℃)", "10分钟最大风速(m/s)", "最大风速时风向"]]
        results.append(corr.values)
    low = np.percentile(results, 2.5, axis=0)
    up = np.percentile(results, 97.5, axis=0)
    DataFrame(data=list(zip(low, up)), index=["负荷", "气温", "风速", "风向"]).to_csv("./变压器油温分析.csv", index=True, header=None)

    # (3) 请根据给出的油温和负荷数据（变压器油温预测数据集_测试集.csv）为不同的变压器构建油温预测模型。
    # 预测值为py，真实值为y，评价指标采用均方误差（mse）。最终结果按照下表的参考文件结构保存为变压器油温预测.csv。
    data_test = pd.read_csv("./变压器油温预测数据集_测试集.csv", encoding="gbk")
    data_test = pre_processing(data_test)
    x_train, onehot = feature_engineering(oil_temperature, None)
    x_test, _ = feature_engineering(data_test, onehot)
    y_train = x_train.pop("remainder__油温(℃)")
    y_test = x_test.pop("remainder__油温(℃)")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = Series(model.predict(x_test), index=x_test.index, name="pred__油温(℃)")
    data_pred = pd.concat([x_test, y_test, y_pred], axis=1)
    trans_x = data_pred[data_pred["onehot__变电站名称_X01变电站"] == 1]
    trans_y = data_pred[data_pred["onehot__变电站名称_Y01变电站"] == 1]
    trans_z = data_pred[data_pred["onehot__变电站名称_Z01变电站"] == 1]
    with open("./变压器油温预测.csv", "w", encoding="utf-8") as f:
        f.write("Name,MSE\n")
        f.write(f"X01变电站,{mean_squared_error(trans_x['remainder__油温(℃)'], trans_x['pred__油温(℃)'])}\n")
        f.write(f"Y01变电站,{mean_squared_error(trans_y['remainder__油温(℃)'], trans_y['pred__油温(℃)'])}\n")
        f.write(f"Z01变电站,{mean_squared_error(trans_z['remainder__油温(℃)'], trans_z['pred__油温(℃)'])}\n")
