# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # 1
    df = pd.read_csv("./9.2/data.csv", encoding="utf-8")
    mask = (df < df.mean(axis=0) - 3 * df.std(axis=0)) | (df > df.mean(axis=0) + 3 * df.std(axis=0))
    print(df.index[mask["f1"]])
    print(df.index[mask["f2"]])
    # 2
    df.loc[mask["f1"], "f1"] = np.nan
    df.loc[mask["f2"], "f2"] = np.nan
    df = df.fillna(df.median()).astype(int)
    print(df)