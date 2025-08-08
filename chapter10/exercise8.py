# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./分类1- 家庭用电安全隐患分析/household_electric_power_analysis.csv", na_values="?", dtype=str)
    data = data.fillna("0")
    data["other_active_energy_consumed"] = data["Global_active_power"].astype(float) * 1000 / 60 - data["Sub_metering_1"].astype(float) - data["Sub_metering_2"].astype(float) - data["Sub_metering_3"].astype(float)
    data["other_active_energy_consumed"] = data["other_active_energy_consumed"].apply(lambda x: "%.3f" % round(x, 3))
    data.to_csv("./分类1- 家庭用电安全隐患分析/1-1.csv", index=False)
    # 2
    df2 = data[data["Date"] == "1/10/2007"].copy()
    df2["DateTime"] = pd.to_datetime(df2["Date"] + " " + df2["Time"], format="%d/%m/%Y %H:%M:%S")
    df2 = df2.iloc[:, 2:]
    df2["hour"] = df2["DateTime"].apply(lambda x: x.hour)
    df2["Global_active_power"] = df2["Global_active_power"].astype(float)
    result2 = df2[["hour", "Global_active_power"]].groupby("hour").mean()
    plt.figure(2)
    plt.plot(result2.index, result2["Global_active_power"])
    plt.xlabel("Hour")
    plt.ylabel("Kilo Watt")
    plt.title("Global_active_power")
    plt.savefig("./分类1- 家庭用电安全隐患分析/1-2.png")
    plt.show()
    # 3
    targets = pd.read_csv("./分类1- 家庭用电安全隐患分析/train_label.csv")
    targets["DateTime"] = pd.to_datetime(targets["Date"] + " " + targets["Time"])
    targets["hour"] = targets["DateTime"].apply(lambda x: x.hour)
    result3 = targets["hour"].value_counts().iloc[0:3].to_frame().rename(columns={"hour": "Count"})
    result3.index.name = "Hour"
    result3.to_csv("./分类1- 家庭用电安全隐患分析/1-3.csv", index=True)
    # 4
    data["DateTime"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%d/%m/%Y %H:%M:%S")
    data = data.drop(columns=["Date", "Time"])
    x_train = data[(data["DateTime"] >= "2007-1-1") & (data["DateTime"] < "2008-7-1")].set_index("DateTime").astype(float)
    x_test = data[(data["DateTime"] >= "2007-7-1") & (data["DateTime"] < "2009-1-1")].set_index("DateTime").astype(float)
    y_train = targets.set_index("DateTime").drop(columns=["Date", "Time"]).reindex(x_train.index).fillna(0)["hour"]
    y_train[y_train > 0] = 1
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_series = pd.Series(y_pred, index=x_test.index)
    result4 = y_pred_series[y_pred_series > 0].to_frame()
    result4["DateTime"] = result4.index
    result4["Date"] = result4["DateTime"].apply(lambda x: x.strftime("%Y/%m/%d"))
    result4["Time"] = result4["DateTime"].apply(lambda x: x.strftime("%H:%M:%S"))
    result4[["Date", "Time"]].to_csv("./分类1- 家庭用电安全隐患分析/1-4.csv", index=False)
