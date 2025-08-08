# -*- coding: utf-8 -*-
"""
Grid Disruption.csv文件保存了美国15年停电情况的统计信息。请完成以下问题：
"""
import pandas as pd


if __name__ == '__main__':
    # (1) 分别统计导致停电次数最多的五个事件和导致停电次数最少的五个事件以及对应的次数，将结果保存为文件3-1.csv。
    # 按照次数递减的顺序排列。参考文件结构见下表
    data = pd.read_csv("./Grid_Disruption.csv")
    result = data["Event Description"].value_counts().to_frame().rename(columns={
        "count": "Number"
    }).reset_index()
    result = pd.concat([result.head(5), result.tail(5)], axis=0)
    result.to_csv("./3-1.csv", index=False)
