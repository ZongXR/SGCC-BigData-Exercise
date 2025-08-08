# -*- coding: utf-8 -*-
import pandas as pd


if __name__ == '__main__':
    cur_data = pd.read_csv("./9.5/cur_data.csv")
    vol_data = pd.read_csv("./9.5/vol_data.csv")
    data = cur_data.merge(vol_data, how="outer", on=["CONS_NO", "DATA_DATE", "PHASE_FLAG"])
    data.to_csv("./9.5/cur_vol_data.csv", index=False)