# -*- coding: utf-8 -*-
import pandas as pd


def get_codes(xx: pd.Series, _encoder: dict) -> [int]:
    """
    获取编码\n
    :param xx: 输入一行
    :param _encoder: 编码器
    :return: 输出编码
    """
    _result = xx.index[xx.astype(bool)]
    return [_encoder[k] for k in _result]


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./关联1-高压用户电费回收风险关联分析/电费回收风险.csv", encoding="gbk")
    for column in data.columns:
        if "电费" in column or "电量" in column or "间隔天数" in column:
            data[column] = pd.cut(data[column], bins=5)
    # 2
    social_features = (
        "股权出质总金额",
        "司法冻结金额",
        "动产抵押金额",
        "司法执行金额",
        "工商异常数量",
        "开庭公告数量",
        "裁判文书数量"
    )
    for column in data.columns:
        if column in social_features:
            data[column] = data[column].apply(lambda x: "是" if x > 0 else "否")
    # 3
    data["逾期数量"] = data["逾期数量"].apply(lambda x: "是" if x > 0 else "否")
    # 4
    data = pd.get_dummies(data, columns=data.columns)
    # 5
    encoder = {k: v for k, v in zip(data.columns, range(len(data.columns)))}
    decoder = {v: k for k, v in encoder.items()}
    for i in range(data.shape[0]):
        print(get_codes(data.iloc[i], encoder))
