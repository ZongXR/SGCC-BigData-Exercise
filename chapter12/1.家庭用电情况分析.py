# -*- coding: utf-8 -*-
"""
2017 House Energy Use.csv文件保存了某家庭2017年的用电情况，数据点采集以30分钟间隔进行。样例数据见下表。
"""
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


def get_hour_period(hour) -> str:
    """
    获取小时时间段\n
    :param hour: 小时
    :return: 时间段
    """
    if 11 <= hour <= 14:
        return "11:00-14:00"
    elif 17 <= hour <= 20:
        return "17:00-20:00"
    else:
        return ""


if __name__ == '__main__':
    # (1) 找出数据集中的缺失点，并进行补全。数据点补全方法为全年该时间点（同一时刻，精确到分钟）用电量的算术平均值，数值计算到小数点后第一位。
    # 将计算得到的缺失点的补全结果保存为文件1-1.csv，参考文件结构见下表。数据项按照日期从前到后、时间从早到晚的顺序排列。
    data = pd.read_csv("./2017 House Energy Use.csv", sep="\t")
    data["time"] = data["Date"] + " " + data["Time of day"]
    data["time"] = pd.to_datetime(data["time"])
    data["month_day_hour_minute"] = data["time"].apply(lambda x: f"{x.month}-{x.day} {x.hour}:{x.minute}")
    mean_electricity = data[["month_day_hour_minute", "Electricity Used (kwH)"]].groupby("month_day_hour_minute").mean().reset_index().rename(columns={
        "Electricity Used (kwH)": "mean_electricity"
    })
    result = pd.merge(data, mean_electricity, how="left", on="month_day_hour_minute")
    result["Electricity Used (kwH)"] = result["Electricity Used (kwH)"].fillna(result["mean_electricity"])
    result = data.sort_values("time", ascending=True).dropna()
    result["Date"] = result["time"].apply(lambda x: f"{x.year}/{x.month}/{x.day}")
    result["Electricity Used (kwH)"] = result["Electricity Used (kwH)"].apply(lambda x: round(x, 1))
    result[["Time of day", "Electricity Used (kwH)", "Date"]].to_csv("./1-1.csv", index=False)

    # (2) 将每个月的用电量累加求和，数值计算到小数点后第一位。结果保存为1-2.csv文件，参考文件结构见下表。
    # 将2017年12个月的电量求和画成柱状图，其中横坐标为日期，纵坐标为该月用电量，标题为Electricity Used，保存为文件1-1.png
    data["year_month"] = data["time"].apply(lambda x: f"{x.month}/{x.year}")
    result = data[["year_month", "Electricity Used (kwH)"]].groupby("year_month").sum().reset_index().rename(columns={
        "year_month": "Date"
    })
    result["Electricity Used (kwH)"] = result["Electricity Used (kwH)"].apply(lambda x: round(x, 1))
    result["Date"] = pd.to_datetime(result["Date"], format="%m/%Y")
    result = result.sort_values("Date", ascending=True)
    result["Date"] = result["Date"].apply(lambda x: f"{x.month}/{x.year}")
    result[["Electricity Used (kwH)", "Date"]].to_csv("./1-2.csv", index=False)
    plt.figure(1)
    plt.bar(result["Date"], result["Electricity Used (kwH)"])
    plt.xlabel('Date')
    plt.ylabel('Electricity Used (kwH)')
    plt.xticks(rotation=60)
    plt.title("Electricity Used")
    plt.savefig("./1-1.png")
    plt.show()

    # (3) 选取10月份数据按照日期从前到后、时间从早到晚的顺序重新排列，并保存为文件1-3.csv。参考文件结构见（1）中的表。
    result = data[data["time"].apply(lambda x: x.month == 10)]
    result = result.sort_values("time", ascending=True)
    result[["Time of day", "Electricity Used (kwH)", "Date"]].to_csv("./1-3.csv", index=False)

    # (4) 对10月份每天11:00-14:00间6个时间点和17:00-20:00间6个时间点的用电量分别求和，将结果以两条折线的形式在一张图上表现出来。
    # 其中，横坐标依次为1、2、…、31，纵坐标为电量值，每条曲线使用不通的颜色进行区分，并添加图例。
    # 每天用电量在折线图中标注出来，图形标题为October Electricity Used，将图形保存为1-2.png
    data["hour_period"] = data["time"].apply(lambda x: get_hour_period(x.hour))
    result = data[(data["time"].apply(lambda x: x.month == 10)) & (data["hour_period"] != "")].copy()
    result["day"] = result["time"].apply(lambda x: x.day)
    result = result[["day", "hour_period", "Electricity Used (kwH)"]].groupby(by=["day", "hour_period"]).sum().reset_index()
    plt.figure(2)
    plt.plot(result[result["hour_period"] == "11:00-14:00"]["day"], result[result["hour_period"] == "11:00-14:00"]["Electricity Used (kwH)"], label="11:00-14:00")
    plt.plot(result[result["hour_period"] == "17:00-20:00"]["day"], result[result["hour_period"] == "17:00-20:00"]["Electricity Used (kwH)"], label="17:00-20:00")
    for x, y in zip(result[result["hour_period"] == "11:00-14:00"]["day"], result[result["hour_period"] == "11:00-14:00"]["Electricity Used (kwH)"]):
        plt.text(x, y, y, ha="center", va="bottom", fontsize=10)
    for x, y in zip(result[result["hour_period"] == "17:00-20:00"]["day"], result[result["hour_period"] == "17:00-20:00"]["Electricity Used (kwH)"]):
        plt.text(x, y, y, ha="center", va="bottom", fontsize=10)
    plt.title("October Electricity Used")
    plt.legend()
    plt.savefig("./1-2.png")
    plt.show()

    # (5) 查找10月份的用电数据，依次找出用电量最多、用电量最少的日期，用电量最大、用电量最小的时间节点，
    # 10月份时间节点用电的算数平均数、中位数、方差（结果精确到小数点后第4位），保存结果为文件1-4.csv，参考文件结构见下表。
    result = data[data["time"].apply(lambda x: x.month == 10)].copy()
    usage_sum_by_date: DataFrame = result[["Date", "Electricity Used (kwH)"]].groupby("Date").sum()
    ser = [
        usage_sum_by_date["Electricity Used (kwH)"].idxmax(),
        usage_sum_by_date["Electricity Used (kwH)"].idxmin(),
        result.set_index("time")["Electricity Used (kwH)"].idxmax().strftime("%m/%d/%Y %H:%M"),
        result.set_index("time")["Electricity Used (kwH)"].idxmin().strftime("%m/%d/%Y %H:%M"),
        result["Electricity Used (kwH)"].mean(),
        result["Electricity Used (kwH)"].median(),
        result["Electricity Used (kwH)"].var(),

    ]
    Series(data=ser, index=[
        "用电量最多的日期", "用电量最少的日期", "用电量最大的时间节点",
        "用电量最少的时间节点", "算术平均数", "中位数", "方差"
    ]).to_csv("./1-4.csv", index=True, header=False)