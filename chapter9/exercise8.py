# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    # 1
    df = pd.read_csv("./9.3/example.data")
    df["Salary"] = (df["Salary"] - df["Salary"].min()) / (df["Salary"].max() - df["Salary"].min())
    print(df)
    # 2
    df["Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())
    print(df)