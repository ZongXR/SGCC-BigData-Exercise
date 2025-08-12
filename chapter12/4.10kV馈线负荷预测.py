# -*- coding: utf-8 -*-
"""
10kV馈线将变电站的电能向末端用户进行输送和分配，是大电网的神经末梢。对10kV馈线的负荷进行预测，有助于了解馈线运行状况，并为负荷转供、线损优化等运行方式安排提供辅助决策。
数据集情况：10kV馈线符合数据主要取自地调D5000系统，气象数据来源于电网气象灾害监测预警系统，书体数据来源系统、数据字段、数据取直等情况见下表。
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def get_season(_date_) -> int:
    """
    获取季节\n
    :param _date_: 日期
    :return: 季节
    """
    if _date_.month in (1, 2, 3):
        return 1
    elif _date_.month in (4, 5, 6):
        return 2
    elif _date_.month in (7, 8, 9):
        return 3
    elif _date_.month in (10, 11, 12):
        return 4
    else:
        raise ValueError("月份不正确")


def is_national_day(_date_) -> bool:
    """
    是否在国庆节期间\n
    :param _date_: 日期
    :return: 是否
    """
    return _date_.date() in [x.to_pydatetime().date() for x in list(pd.date_range("2018-10-01", "2018-10-07"))]


def is_spring_festival(_date_) -> bool:
    """
    是否是春节\n
    :param _date_: 日期
    :return: 是否
    """
    return _date_.date() in [x.to_pydatetime().date() for x in list(pd.date_range("2018-02-15", "2018-02-21"))]


def feature_engineering(df) -> DataFrame:
    """
    特征工程\n
    :param df: 输入数据
    :return: 输出数据
    """
    data = df.copy()
    data["month"] = data["数据时间"].apply(lambda x: x.month)
    data["day"] = data["数据时间"].apply(lambda x: x.day)
    data["hour"] = data["数据时间"].apply(lambda x: x.hour)
    data["minute"] = data["数据时间"].apply(lambda x: x.minute)
    data["地区"] = data["地区"].map({
        "X地市": 0, "Y地市": 1
    })
    data["线路名称"] = data["线路名称"].map({
        "10kVX01医院线": 0, "学校Y01线": 1
    })
    data["线路类型"] = data["线路类型"].map({
        "医院": 0, "学校": 1
    })
    data["母线名称"] = data["母线名称"].map({
        "10kVⅣ段母线": 0, "10千伏Ⅳ段母线": 1
    })
    return data.drop(columns=["数据时间"]).fillna(0)


if __name__ == '__main__':
    # (1) 请根据给出的馈线负荷（10kV馈线负荷预测_训练集.csv）计算2018年各馈线的各月份平均电流、最高电流、最低电流峰谷差等指标，生成一张新的DataFrame表，
    # 并将其保存为10kV馈线负荷预测指标分析.csv，参考格式见下表。
    data = pd.read_csv("./10kV馈线负荷预测_训练集.csv", encoding="gbk")
    data["数据时间"] = pd.to_datetime(data["数据时间"])
    data["Month"] = data["数据时间"].apply(lambda x: x.strftime("%Y%m"))
    result = data[["Month", "电流值"]].groupby("Month").agg(["mean", "max", "min"])
    result.columns = result.columns.droplevel(0)
    result["电流峰谷差"] = result["max"] - result["min"]
    result = result.rename(columns={
        "mean": "月份平均电流",
        "max": "最高电流",
        "min": "最低电流"
    })
    result.to_csv("./10kV馈线负荷预测指标分析.csv", index=True)

    # (2) 请根据给出的馈线负荷，简要分析不同季节（1-3、4-6、7-9、10-12月四个季节）和重大节假日 国庆（10.1-10.7）、春节（2018.2.15-2018.2.21）馈线的负荷的可视化。
    # 主要包括：①分别绘制各季节、国庆、春节期间的折线图，其中，各季节折线图的横坐标为季度（1、2、3、4），国庆和春节折线图的横坐标为日期（如：20181001）；
    # 折线图标题分别为season、national-day、new-year，最后分别保存为season.png、national-day.png、new-year.png。
    # ②非春节、国庆的负荷统计量的箱线图，标题为non-holiday，最后保存为non-holiday.png。
    data["season"] = data["数据时间"].apply(get_season)
    data["is_national_day"] = data["数据时间"].apply(is_national_day)
    data["is_spring_festival"] = data["数据时间"].apply(is_spring_festival)
    result_season = data[["season", "电流值"]].groupby("season").mean()
    plt.figure(1)
    result_season["电流值"].plot()
    plt.title("season")
    plt.xticks(result_season.index, result_season.index)
    plt.savefig("./season.png")
    plt.show()

    result_national_day = data[data["is_national_day"].astype(int) == 1][["数据时间", "电流值"]].copy()
    result_national_day["date"] = result_national_day["数据时间"].apply(lambda x: x.strftime("%Y%m%d"))
    result_national_day = result_national_day[["date", "电流值"]].groupby("date").mean()
    plt.figure(2)
    result_national_day["电流值"].plot()
    plt.title("national-day")
    plt.savefig("./national-day.png")
    plt.show()

    result_spring_festival = data[data["is_spring_festival"].astype(int) == 1][["数据时间", "电流值"]].copy()
    result_spring_festival["date"] = result_spring_festival["数据时间"].apply(lambda x: x.strftime("%Y%m%d"))
    result_spring_festival = result_spring_festival[["date", "电流值"]].groupby("date").mean()
    plt.figure(3)
    result_spring_festival["电流值"].plot()
    plt.title("new-year")
    plt.savefig("./new-year.png")
    plt.show()

    result_non_holiday = data[(data["is_spring_festival"].astype(int) == 0) & (data["is_national_day"].astype(int) == 0)]
    plt.figure(4)
    result_non_holiday["电流值"].plot(kind="box")
    plt.title("non-holiday")
    plt.savefig("./non-holiday.png")
    plt.show()

    # (3) 请根据给出的馈线负荷（10kV馈线负荷预测_训练集.csv）经过特征工程，构建馈线—气象负荷模型，
    # 并预测2019.1.1-2019.1.3（10kV馈线负荷预测_测试集.csv）三日各个时间段的电流值。预测值为py，真实值为y,评价指标采用均方误差（mse），
    # 结果以MSE=XXXXX的格式保存为text1.txt。
    x_train = feature_engineering(data).drop(columns=["is_national_day", "is_spring_festival", "Month"])
    y_train = x_train.pop("电流值")
    model = GradientBoostingRegressor(n_estimators=1, random_state=42)
    model.fit(x_train, y_train)

    x_test = pd.read_csv("./10kV馈线负荷预测_测试集.csv", encoding="gbk")
    x_test["数据时间"] = pd.to_datetime(x_test["数据时间"])
    x_test["season"] = x_test["数据时间"].apply(get_season)
    x_test = feature_engineering(x_test)
    y_test = x_test.pop("电流值")
    y_pred = model.predict(x_test)
    with open("./text1.txt", "w", encoding="gbk") as f:
        f.write(f"MSE={mean_squared_error(y_test, y_pred)}")
