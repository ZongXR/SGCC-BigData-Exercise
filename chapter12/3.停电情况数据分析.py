# -*- coding: utf-8 -*-
"""
Grid Disruption.csv文件保存了美国15年停电情况的统计信息。请完成以下问题：
"""
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def to_int(number):
    """
    转化为int类型\n
    :param number: 数字
    :return: 整型
    """
    try:
        return int(number)
    except ValueError as e:
        return np.nan


if __name__ == '__main__':
    # (1) 分别统计导致停电次数最多的五个事件和导致停电次数最少的五个事件以及对应的次数，将结果保存为文件3-1.csv。
    # 按照次数递减的顺序排列。参考文件结构见下表
    data = pd.read_csv("./Grid_Disruption.csv")
    result = data["Event Description"].value_counts().to_frame().rename(columns={
        "count": "Number"
    }).reset_index()
    result = pd.concat([result.head(5), result.tail(5)], axis=0)
    result.to_csv("./3-1.csv", index=False)

    # (2) 计算2000-2014年各年累计的停电天数（不满一天的按一天计算），将结果保存为文件3-2.csv，参考文件结构见下表。
    # 画出对应的饼状图，其中标题为Grid Disruption time，并标出对应的停电天数
    data[data == "Unknown"] = np.nan
    data["start time"] = data["Date Event Began"] + " " + data["Time Event Began"]
    data["end time"] = data["Date of Restoration"] + " " + data["Time of Restoration"]
    data["start time"] = pd.to_datetime(data["start time"], format="%m/%d/%Y %I:%M %p", errors="coerce")
    data["end time"] = pd.to_datetime(data["end time"], format="%m/%d/%Y %I:%M %p", errors="coerce")
    data.loc[(data["Date of Restoration"] == "Ongoing") | (data["Time of Restoration"] == "Ongoing"), "end time"] = data["end time"].max()
    df = data.dropna().copy()
    df["start date"] = df["start time"].dt.date
    df["end date"] = df["end time"].dt.date
    result = set()
    for _, row in df.iterrows():
        date_range = pd.date_range(row['start date'], row['end date'])
        result.update(date_range)
    years = Series([x.year for x in result])
    years.value_counts().to_csv("./3-2.csv", header=["Time"], index=True, index_label="Year")
    plt.figure(1)
    years.value_counts().plot(kind="pie")
    plt.title("Grid Disruption time")
    plt.show()

    # (3) 找出2000-2014年期间停电时间最长的十次停电事件（时间精确到小时），
    # 并按照时间递减的顺序将结果保存到文件3-3.csv文件中，每条事件的内容与题目所给数据一致。
    data["during time"] = data["end time"] - data["start time"]
    data = data.sort_values(by="during time", ascending=False)
    data.drop(columns=["start time", "end time", "during time"]).to_csv("./3-3.csv", index=False)

    # (4) 计算在2000-2014年每个事件的Demand Loss和Number of Customers Affected 两列算术平均值，其中没有具体数据的不列入计算范围。
    # 将结果保存到文件3-4.csv中，事件顺序按照字母序升序排列。参考文件结构见下表。
    data["Demand Loss"] = data["Demand Loss (MW)"].apply(to_int)
    data["Number of Customers Affected"] = data["Number of Customers Affected"].apply(to_int)
    result = data[["Event Description", "Demand Loss", "Number of Customers Affected"]].groupby("Event Description").mean()
    result.to_csv("./3-4.csv", index=True)

    # (5) 将该表Demand Loss和Number of Customers Affected 两列的异常值（N/A，-，Unknown）
    # 分别补全，补全策略如下：①如果异常值所在的事件列具有有效值，则使用算术平均值补全；②如果异常值所在的事件列没有有效值，则使用0补全。
    # 补全后的Demand Loss和Number of Customers Affected两列单独保存为文件3-5.csv，顺序与原数据保持一致，且保持的格式全部为浮点数
    # （保留到小数点后第三位）。参考文件结构见下表。
    result = result.fillna(0).round(3)
    df = pd.merge(data, result, how="left", left_on="Event Description", right_index=True, suffixes=("", "_mean"))
    df["Demand Loss"] = df["Demand Loss"].fillna(df["Demand Loss_mean"])
    df["Number of Customers Affected"] = df["Number of Customers Affected"].fillna(df["Number of Customers Affected_mean"])
    df[["Demand Loss", "Number of Customers Affected"]].round(3).to_csv("./3-5.csv", index=False)

    # (6) 使用补全后的Demand Loss和Number of Customers Affected 两列数据。将全部数据划分为五类，编号1-5。
    # 给出每类的事件数量以及数量最多的事件，将结果保存为文件3-6.csv，参考文件结构见下表。
    # 以Demand Loss为横坐标，Number of Customers Affected为纵坐标，将聚类结果以散点图的形式表现出来。
    # 要求每个类别使用不同的颜色进行区分，添加图例，图形标题为Cluster Result，将图片保存为3-1.png。
    scaler = StandardScaler()
    x = scaler.fit_transform(df[["Demand Loss", "Number of Customers Affected"]])
    model = KMeans(n_clusters=5, random_state=42)
    y_pred = model.fit_predict(x)
    result = df[["Event Description", "Demand Loss", "Number of Customers Affected"]].copy()
    result["No"] = y_pred
    events = result.groupby("No")["Event Description"].apply(lambda x: x.mode()[0]).reset_index()
    result = pd.merge(result, events, how="left", left_on="No", right_on="No", suffixes=("", "_pred"))
    result = result[["No", "Event Description_pred", "Demand Loss", "Number of Customers Affected"]].copy()
    counts = result["No"].value_counts().to_frame().rename(columns={
        "count": "Number of Disruption"
    })
    result1 = result[["No", "Event Description_pred"]].drop_duplicates().rename(columns={
        "Event Description_pred": "Event Description"
    })
    result2 = pd.merge(counts, result1, how="left", left_index=True, right_on="No")
    result2[["No", "Number of Disruption", "Event Description"]].to_csv("./3-6.csv", index=False)

    plt.figure(2)
    colors = ["r", "g", "b", "y", "k"]
    for i in result["No"].unique():
        df = result[result["No"] == i]
        plt.scatter(df["Demand Loss"], df["Number of Customers Affected"], color=colors[i], label=df["Event Description_pred"].mode()[0])
    plt.legend()
    plt.title("Cluster Result")
    plt.savefig("./3-1.png")
    plt.show()
