# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("./9.5/Italy Solar Generation and Demand.csv")
    data["utc_timestamp"] = pd.to_datetime(data["utc_timestamp"])
    print(data.isnull().sum(axis=0))
    data["year-month"] = data["utc_timestamp"].apply(lambda x: x.year * 100 + x.month)
    values = data[["year-month", "Demand"]].groupby("year-month").transform(lambda x: x.mean())
    data["Demand"] = data["Demand"].fillna(values["Demand"])
    print(data.isnull().sum(axis=0))