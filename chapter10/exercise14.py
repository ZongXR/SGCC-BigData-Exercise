# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor


def get_error_type_count(_faults: pd.DataFrame) -> pd.Series:
    """
    获取每个月的最多故障类型及数量\n
    :param _faults: 每个月的故障数据
    :return: 故障类型, 数量
    """
    _result = _faults["ERR_TYPE"].value_counts(ascending=False)
    return pd.Series({
        "TOP_COUNT_TYPE": _result.index[0],
        "COUNT": _result.iloc[0]
    })


if __name__ == '__main__':
    # 1
    data_error = pd.read_csv("./回归2 - 电网故障与气象关联分析/gridfaults_weather_analysis_error.csv")
    data_error["DATA_DATE_E"] = pd.to_datetime(data_error["DATA_DATE_E"])
    data_error["MONTH"] = data_error["DATA_DATE_E"].apply(lambda x: x.month)
    result1 = data_error[data_error["DATA_DATE_E"].apply(lambda x: x.year == 2014)].groupby("MONTH").apply(lambda x: get_error_type_count(x))
    result1.loc[[7, 8, 9], :].to_csv("./回归2 - 电网故障与气象关联分析/1_1.csv", encoding="utf-8")
    # 2
    data_weather = pd.read_csv("./回归2 - 电网故障与气象关联分析/gridfaults_weather_analysis_weather.csv")
    data_weather["DATA_DATE_T"] = pd.to_datetime(data_weather["DATA_DATE_T"])
    data_weather["day"] = data_weather["DATA_DATE_T"].apply(lambda x: x.day + 100 * x.month + 10000 * x.year)
    data_error["day"] = data_error["DATA_DATE_E"].apply(lambda x: x.day + 100 * x.month + 10000 * x.year)
    fault_count: DataFrame = data_error.groupby("day").count()["ERR_TYPE"].to_frame()
    weathers: DataFrame = data_weather[["day", "DATA_R", "DATA_W"]].groupby("day").mean()
    weather_faults = weathers.merge(fault_count, how="inner", left_index=True, right_index=True)
    result3 = DataFrame([[0], [0]], index=["雨量与故障数量之间的关联系数", "风速与故障数量之间的关联系数"], columns=["CORR"])
    result3.index.name = "RELATION"
    result3.loc["雨量与故障数量之间的关联系数", "CORR"] = round(pearsonr(weather_faults["DATA_R"], weather_faults["ERR_TYPE"])[0], 2)
    result3.loc["风速与故障数量之间的关联系数", "CORR"] = round(pearsonr(weather_faults["DATA_W"], weather_faults["ERR_TYPE"])[0], 2)
    result3.to_csv("./回归2 - 电网故障与气象关联分析/1_2.csv", index=True, encoding="utf-8")
    # 3
    data_label = pd.read_csv("./回归2 - 电网故障与气象关联分析/gridfaults_weather_analysis_train_label.csv")
    data_label["DATA_DATE_E"] = pd.to_datetime(data_label["DATA_DATE_E"])
    data_weather["DATA_P"] = data_weather["DATA_P"].apply(lambda x: x if isinstance(x, float) else float(x.replace(",", "")))
    data_train = data_weather.merge(data_label, how="inner", left_on="DATA_DATE_T", right_on="DATA_DATE_E")
    data_train = data_train.drop(columns=["DATA_DATE_T", "day", "DATA_DATE_E"])
    x_train = data_train.drop(columns=["ERR_NUM"])
    y_train = data_train["ERR_NUM"].copy()
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(x_train, y_train)
    data_test = pd.read_csv("./回归2 - 电网故障与气象关联分析/gridfaults_weather_analysis_test.csv")
    data_test["DATA_DATE_E"] = pd.to_datetime(data_test["DATA_DATE_E"])
    data_test = data_test.merge(data_weather, how="inner", left_on="DATA_DATE_E", right_on="DATA_DATE_T")
    x_test = data_test.drop(columns=["DATA_DATE_E", "DATA_DATE_T", "day", "ERR_NUM"]).reindex(columns=x_train.columns)
    y_pred = model.predict(x_test)
    data_test["ERR_NUM"] = y_pred
    data_test[["DATA_DATE_E", "ERR_NUM"]].set_index("DATA_DATE_E").astype(int).to_csv("./回归2 - 电网故障与气象关联分析/1_3.csv", index=True, date_format="%Y/%m/%d", encoding="utf-8")