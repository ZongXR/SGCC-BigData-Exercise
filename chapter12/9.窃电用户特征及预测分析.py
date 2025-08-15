# -*- coding: utf-8 -*-
"""
有效管理社会生产生活用电、提高电能利用率是现代化社会与电力企业共同关注的问题。就我国用电相关的纠纷案件来说，比较突出的问题就是违约用电与窃电问题，
窃电手段不断变化发展，使得窃电范围不断扩大，通过分析窃电用户特征预测潜在窃电风险，具有一定的经济意义。
假设已知某地区2015年1月1日-2015年12月31日每个用户的每日用电情况数据stolen_electric_analysis_1.csv。数据说明见下表。
已知该地区部分用户的窃电情况标签表stolen_electric_analysis_2.csv中标号1为确定窃电用户、标号0为不确定的窃电用户。
其中每个用户每日用电情况表和部分用户窃电情况标签表之间用CONS_NO字段关联，每个用户每日用电情况表含有每个用户每天的用电数据，
表中每个用户CONS_NO可占多行数据，窃电情况标签表中每个用户只占1行数据。
stolen_electric_analysis_3.csv为测试数据集，包含所有待判别用户。数据说明见下表。
"""
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from zipfile import ZipFile
from sklearn.ensemble import GradientBoostingClassifier


pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = ['SimHei']           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False             # 用来正常显示负号


def get_start_end(_df_: DataFrame) -> Series:
    """
    获得其实日期、终止日期以及对象的电量\n
    :param _df_: 输入数据
    :return: 输出数据
    """
    _df_ = _df_.sort_values("DT", ascending=True)
    _result_ = {
        "START_DATE": _df_["DT"].min(),
        "END_DATE": _df_["DT"].max(),
        "TOTAL_START_KWH": round(_df_["KWH_BEGIN"].iloc[0], 2),
        "TOTAL_END_KWH": round(_df_["KWH_END"].iloc[-1], 2)
    }
    return Series(_result_)


def feature_engineering(_df_: DataFrame) -> DataFrame:
    """
    特征工程\n
    :param _df_: 输入数据
    :return: 输出数据
    """
    _result_ = _df_.copy()
    _result_["START_DATE"] = pd.to_datetime(_result_["START_DATE"], format="%Y%m%d")
    _result_["END_DATE"] = pd.to_datetime(_result_["END_DATE"], format="%Y%m%d")
    _result_["days"] = (_result_["END_DATE"] - _result_["START_DATE"]).apply(lambda x: x.days)
    return _result_.drop(columns=["START_DATE", "END_DATE"]).set_index("CONS_NO")


if __name__ == '__main__':
    # (1) 请根据给出的用户每日用电情况数据（stolen_electric_analysis_1.csv），将每个用户的用电情况按照日期进行排序。
    # 取用户CONS_NO为1652714033的用电数据，绘制该用户有记录的日期的用电量（kWh）的折线图：横坐标为精确到天的日期，名称为DATE，纵坐标为kWh数值，名称为KWH；
    # 标题为CONS_NO=1652714033。其中折线图保存为9-1.png，程序保存为9-1.py。
    with ZipFile("./stolen_electric_analysis_1.zip", "r") as zip_file:
        with zip_file.open("stolen_electric_analysis_1.csv") as f:
            data1 = pd.read_csv(f)
    result1 = data1[data1["CONS_NO"] == 1652714033].sort_values("DT")
    result1 = result1.rename(columns={
        "DT": "DATE"
    })
    result1["DATE"] = result1["DATE"].astype(str)
    plt.figure(1)
    result1[["DATE", "KWH"]].set_index("DATE")["KWH"].plot()
    plt.ylabel("KWH")
    plt.title("CONS_NO=1652714033")
    plt.savefig("./9-1.png")
    plt.show()

    # (2) 提取用户每日用电情况数据（stolen_electric_analysis_1.csv），删除KWH_BEGIN、KWH_END、KWH任意字段缺失的整行数据，
    # 数据结构应与源文件一致（需保留表头）。其中程序保存为9-2.py，计算的结果保存为文件9-2.csv。
    result2 = data1.dropna(how="any", subset=["KWH_BEGIN", "KWH_END", "KWH"])
    result2.to_csv("./9-2.csv", index=False)

    # (3) 根据（2）中得到的数据结果文件9-2.csv，计算该地区的潜在窃电用户表（stolen_electric_analysis_2.csv）中所给用户的有效开始时间（START_DATE）、
    # 有效开始时间当天起始用电量（TOTAL_START_KWH）、有效结束时间（END_DATE）、有效结束时间当天终止用电量（TOTAL_END_KWH）、总用电量（SUM_KWH），
    # 其中总用电量=有效结束时间当天终止用电量-有效开始时间当天其实用电量，结果保留小数点后两位。程序保存为9-3.py，计算的结果保存为文件9-3.csv，参考文件结构见下表。
    data2 = pd.read_csv("./stolen_electric_analysis_2.csv")
    data_all = result2.groupby("CONS_NO").apply(get_start_end)
    data_all["START_DATE"] = data_all["START_DATE"].astype(int)
    data_all["END_DATE"] = data_all["END_DATE"].astype(int)
    result3 = pd.merge(data_all, data2, how="inner", left_index=True, right_on="CONS_NO")
    result3["SUM_KWH"] = result3["TOTAL_END_KWH"] - result3["TOTAL_START_KWH"]
    result3 = result3.reindex(columns=["CONS_NO", "START_DATE", "TOTAL_START_KWH", "END_DATE", "TOTAL_END_KWH", "SUM_KWH", "CHK_STATE"])
    result3.drop(columns=["CHK_STATE"]).to_csv("./9-3.csv", index=False)

    # (4) 基于stolen_electric_analysis_1.csv和stolen_electric_analysis_2.csv文件中2015年1月1日-2015年12月31日的数据，设计一个二分类模型。
    # 该模型包含窃电用户和不确定窃电用户两类，并对stolen_electric_analysis_3.csv中所有待判别用户进行分析，得出判别结果（1表示窃电，0表示未知）。
    # 其中程序保存为9-4.py，计算的结果保存为文件9-4.csv，参考文件结构见下表。
    x_train = feature_engineering(result3)
    y_train = x_train.pop("CHK_STATE")
    data3 = pd.read_csv("./stolen_electric_analysis_3.csv")
    x_test = pd.merge(data_all, data3, how="inner", left_index=True, right_on="CONS_NO")
    x_test["SUM_KWH"] = x_test["TOTAL_END_KWH"] - x_test["TOTAL_START_KWH"]
    x_test = x_test.reindex(columns=["CONS_NO", "START_DATE", "TOTAL_START_KWH", "END_DATE", "TOTAL_END_KWH", "SUM_KWH", "CHK_STATE"])
    x_test = feature_engineering(x_test)
    y_test = x_test.pop("CHK_STATE")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = Series(model.predict(x_test), index=x_test.index)
    y_pred.to_csv("./9-4.csv", header=["CHK_STATE"])
