# -*- coding: utf-8 -*-


def var(lst: list) -> float:
    """
    计算方差\n
    :param lst: 输入列表
    :return: 方差
    """
    mean = sum(lst) / len(lst)
    return sum(list(map(lambda x: (x - mean) ** 2, lst))) / len(lst)


if __name__ == '__main__':
    print(var([2, 3, 4]))