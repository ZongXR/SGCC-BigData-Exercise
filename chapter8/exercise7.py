# -*- coding: UTF-8 -*-
import pandas as pd


if __name__ == '__main__':
    # 1
    df = pd.read_csv("./9.2/example.csv")
    df1 = df.drop_duplicates()
    print(df1)
    # 2
    df2 = df.drop_duplicates(subset=["Salary"])
    print(df2)
    # 3
    df3 = df.drop_duplicates(subset=["ID", "Salary", "Age"])
    print(df3)