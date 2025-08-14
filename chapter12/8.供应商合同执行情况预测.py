# -*- coding: utf-8 -*-
"""
电网每年都需要和很多设备供应商有相关业务来往，因此可通过分析与供应商的业务来往数据来保证物资供应结算的及时性和准确性。
本题数据提供了某省2018年7月份的合同执行数据，数据来自电网ERP系统，其中样例数据见下表。
"""
from typing import Optional
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def preprocessing(_df_: DataFrame) -> DataFrame:
    """
    特征工程\n
    :param _df_: 输入数据
    :return: 输出数据, 编码器
    """
    _result_ = _df_.copy()
    _result_["合同金额"] = _result_["合同金额"].apply(lambda x: float(x.replace(",", "")))
    _result_["已付总金额"] = _result_["已付总金额"].apply(lambda x: float(x.replace(",", "")))
    _result_["已付预付款"] = _result_["已付预付款"].apply(lambda x: float(x.replace(",", "")))
    _result_["已付到货款"] = _result_["已付到货款"].apply(lambda x: float(x.replace(",", "")))
    _result_["已付投运款"] = _result_["已付投运款"].apply(lambda x: float(x.replace(",", "")))
    _result_["已付质保款"] = _result_["已付质保款"].apply(lambda x: x if type(x) is float else float(x.replace(",", "")))
    _result_["付款比例1"] = _result_["付款比例"].apply(lambda x: float(x.split(":")[0]))
    _result_["付款比例2"] = _result_["付款比例"].apply(lambda x: float(x.split(":")[1]))
    _result_["付款比例3"] = _result_["付款比例"].apply(lambda x: float(x.split(":")[2]))
    _result_["付款比例4"] = _result_["付款比例"].apply(lambda x: float(x.split(":")[3]))
    return _result_.drop(columns=["合同编号", "付款比例"])


def feature_engineering(_df_: DataFrame, _encoder_: Optional[ColumnTransformer] = None) -> (DataFrame, ColumnTransformer):
    """
    特征工程\n
    :param _df_: 输入数据
    :param _encoder_: 编码器
    :return: 输出数据
    """
    if _encoder_ is None:
        _encoder_ = ColumnTransformer([
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["物料描述", "供应商名称"]),
        ], remainder="passthrough", n_jobs=-1)
        return DataFrame(_encoder_.fit_transform(_df_), columns=_encoder_.get_feature_names_out()), _encoder_
    else:
        return DataFrame(_encoder_.transform(_df_), columns=_encoder_.get_feature_names_out()), _encoder_


if __name__ == '__main__':
    # (1) 请根据给出的合同执行情况数据表（“合同执行情况_训练集.xlsx”），计算出合作供应商公司中合同总额最高的5家公司、采购次数最多的5种物料，
    # 并以csv格式输出，参考文件结构见下表，其中程序保存1.py，输出结果保存供应商合同执行分析1.csv
    data_train = pd.read_excel("./合同执行情况_训练集.xlsx")
    data_train = preprocessing(data_train)
    companies = data_train[["供应商名称", "合同金额"]].groupby("供应商名称").sum().sort_values("合同金额", ascending=False).index[0:5].tolist()
    wuliao = data_train["物料描述"].value_counts(ascending=False)
    DataFrame(list(zip(companies, wuliao.index.tolist()[0:5])), columns=["采购合同总额最高的5家公司", "采购次数最多的5种物料"]).to_csv(
        "./供应商合同执行分析1.csv", index=False
    )

    # (2) 请将采购次数最多的5种物料的采购次数绘制成饼图。标题、每种类别的比例、物料类别名称（yy1、yy2等需替换为具体的物料描述）需有所体现。
    plt.figure(1)
    wuliao.iloc[0:5].plot(kind="pie", autopct='%.2f%%')
    plt.title("采购次数最多的5种物料")
    plt.show()

    # (3) 请使用训练数据（合同执行情况_训练集.xlsx）构建模型，并对测试集（合同执行情况_测试集.xlsx）进行预测，预测该笔交易是否能够按时交货，
    # 评价指标采用AUC进行评估。
    data_test = pd.read_excel("./合同执行情况_测试集（含答案）.xlsx")
    data_test = preprocessing(data_test)
    x_train, encoder = feature_engineering(data_train, None)
    y_train = x_train.pop('remainder__是否按时交货')
    x_test, _ = feature_engineering(data_test, encoder)
    y_test = x_test.pop("remainder__是否按时交货")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)
    print(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
