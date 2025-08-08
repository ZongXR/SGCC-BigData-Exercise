# -*- coding: utf-8 -*-
import pandas as pd


def read_txt(path: str, separate: str) -> [[str]]:
    """
    读取文本至二维列表\n
    :param path: 路径
    :param separate: 分隔符
    :return: 二维列表
    """
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            result.append(line.strip().split(separate))
    return result


if __name__ == '__main__':
    # 1
    df5 = pd.read_csv("./9.1/data1.txt", delimiter='\t', names=["header"])
    print(df5)
    # 2
    print(read_txt("./9.1/data2.txt", separate="/$/"))
