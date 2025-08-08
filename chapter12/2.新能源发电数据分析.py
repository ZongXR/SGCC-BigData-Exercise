# -*- coding: utf-8 -*-
"""
Germany electricity power for 2012-2016.csv文件保存了德国在2012-2016年电能消耗量（Consumption）、风能（Wind）和太阳能（Solar）发电量
样例数据见下表
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


if __name__ == '__main__':
    # (1) 计算风能和太阳能的发电量总和，并对应保存为新的列Wind+Solar，要求数据保存到小数点后第三位，将文件保存为2-1.csv文件。参考文件结构见下表
    data = pd.read_csv("./Germany electricity power for 2012-2016.csv")
    data["Wind+Solar"] = data["Wind"] + data["Solar"]
    data["Wind+Solar"] = data["Wind+Solar"].apply(lambda x: round(x, 3))
    data.to_csv("./2-1.csv", index=False)

    # (2) 将电能消耗量，风能、太阳能发电量以每年为单位分别求和，数值计算到小数点后第三位，并画出柱状图。
    # 其中横坐标为年份，纵坐标为电量值，每个柱状图标明对应的数值，图形标题为Electricity Consumption and New Energy。图形保存为图片2-1.png
    data["year"] = data["Date"].apply(lambda x: int(x[0:4]))
    result = data.drop(columns=["Date"]).groupby("year").sum()
    plt.figure(1)
    ax = result.plot(kind="bar")
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.title("Electricity Consumption and New Energy")
    plt.savefig("./2-1.png")
    plt.show()

    # (3) 计算每天电能消耗量与风能和太阳能发电量之和的差值，找出差值最大的10天，并按照差值由大到小的顺序保存为文件2-2.csv，数值计算到小数点后第三位。
    # 参考文件结构见下表。
    data["Difference"] = data["Consumption"] - data["Wind+Solar"]
    data["Difference"] = data["Difference"].apply(lambda x: round(x, 3))
    result = data.sort_values("Difference", ascending=False).iloc[0:10]
    result[["Date", "Consumption", "Wind+Solar", "Difference"]].to_csv("./2-2.csv", index=False)

    # (4) 计算风能与太阳能发电量两组数据的相关系数，并保存为文件2-3.csv
    with open("./2-3.csv", "w", encoding="utf-8") as f:
        f.write(str(data[["Wind", "Solar"]].corr().iloc[0, 1]))

    # (5) 以月为单位，计算2014、2015年和2016年各月电能消耗量之和，并将结果保存为2-4.csv，参考文件结构见下表。
    # 绘制2014、2015和2016年每月电能消耗量之和的折线图，并计算三个月份中对应月份的最大值和最小值，并将最大值增加100，最小值减小100，绘制修正的最大值和最小值曲线。
    # 图形横坐标为日期，精确到月，纵坐标为电量消耗值。五条曲线以不通的颜色区分，图形标题为Electricity Consumption，图形保存为文件2-2.png
    data["month"] = pd.to_datetime(data["Date"]).apply(lambda x: x.month)
    result = data[["year", "month", "Consumption"]].groupby(by=["year", "month"]).sum().reset_index()
    result = result[result["year"].apply(lambda x: x in (2014, 2015, 2016))]
    df = pd.pivot_table(data=result, index=result["month"], columns=result["year"])
    df.columns = df.columns.droplevel()
    df.to_csv("./2-4.csv", index=False)
    df["max"] = df.max(axis=1)
    df["min"] = df.min(axis=1)
    plt.figure(2)
    df.plot()
    plt.title("Electricity Consumption")
    plt.savefig("./2-2.png")
    plt.show()

    # (6)
    data["day"] = pd.to_datetime(data["Date"]).apply(lambda x: x.day)
    x_train = data[["year", "month", "day", "Wind"]]
    y_train = x_train.pop("Wind")
    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(x_train, y_train)
    data_test = pd.read_csv("./Germany electricity power for 2017 - test answer.csv")
    data_test["year"] = pd.to_datetime(data_test["Date"]).apply(lambda x: x.year)
    data_test["month"] = pd.to_datetime(data_test["Date"]).apply(lambda x: x.month)
    data_test["day"] = pd.to_datetime(data_test["Date"]).apply(lambda x: x.day)
    x_test = data_test[["year", "month", "day", "Wind"]]
    y_test = x_test.pop("Wind")
    y_pred = model.predict(x_test)
    data_test["y_pred"] = y_pred
    data_test = data_test[(data_test["year"] == 2017) & (data_test["month"].apply(lambda x: x in (1, 2, 3)))]
    result = data_test[["month", "y_pred"]].groupby("month").sum()
    result = result.rename(index={x: f"2017/{x}" for x in result.index}).T
    result.to_csv("./2-5.csv", index=False)
