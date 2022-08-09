# -*- coding: utf-8 -*-
from pandas import DataFrame


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
    df = list(map(lambda x: ",".join(x), read_txt("./data1.txt", "/$/")))
    with open("./result3_1.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(df))
    # 2
    df2 = DataFrame(data=[
        [85, 96, 93, "NULL"],
        [97, 73, "NULL", 89]
    ], index=[1004, 1023], columns=["语文", "数学", "外语", "历史"])
    df2.index.name = "id"
    df2.to_csv("./result3_2.csv")