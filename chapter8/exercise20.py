# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("./9.5/Germany electricity power for 2012-2016.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data["year"] = data["Date"].apply(lambda x: x.year)
    result = data.groupby("year").apply(lambda x: x.nlargest(5, columns=["Consumption"]))
    print(result.droplevel(0))