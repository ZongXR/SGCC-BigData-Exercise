# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    # 1
    df1 = pd.read_csv("./data1.csv")
    print(df1)
    # 2
    df2 = pd.read_csv("./data2.csv", header=None)
    df2.columns = df1.columns
    print(df2)

