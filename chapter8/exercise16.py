# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    df1 = pd.DataFrame({
        "key": ['a', 'e', 'b', 'a', 'c', 'a', 'b'],
        "data1": range(7)
    })
    df2 = pd.DataFrame({
        "key": ['a', 'b', 'c', 'f'],
        "data2": range(4)
    })
    # 1
    print(df1.merge(df2, how="inner", on="key"))
    # 2
    print(df1.merge(df2, how="outer", on="key"))
    # 3
    print(df1.merge(df2, how="left", on="key"))