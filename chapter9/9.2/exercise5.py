# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    # 1
    df1 = pd.read_csv("./data.csv", encoding="utf-8")
    print(df1.isnull().sum(axis=0).astype(bool))
    # 2
    df10 = df1.copy()
    df10.dropna(axis=0, inplace=True)
    print(df10)
    # 3
    df1.fillna(df1.mean(axis=0), inplace=True)
    print(df1)