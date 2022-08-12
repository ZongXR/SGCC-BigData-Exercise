# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_total_twh(xx: pd.DataFrame) -> pd.Series:
    """
    获取起止日期, 起止日期对应的KWH\n
    :param xx: 输入数据
    :return: 输出
    """
    min_date = xx["DT"].min()
    max_date = xx["DT"].max()
    start_kwh = xx[xx["DT"] == min_date]["KWH_BEGIN"].iloc[0]
    end_kwh = xx[xx["DT"] == max_date]["KWH_END"].iloc[0]
    return pd.Series({
        "START_DATE": min_date,
        "END_DATE": max_date,
        "TOTAL_START_KWH": start_kwh,
        "TOTAL_END_KWH": end_kwh
    })


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./分类2 - 窃电用户特征及预测分析/stolen_electric_analysis_1.csv", dtype=str)
    data = data.dropna(axis=0, how="any", subset=["KWH_BEGIN", "KWH_END", "KWH"])
    data.to_csv("./分类2 - 窃电用户特征及预测分析/1-1.csv", index=False)
    # 2
    df2 = pd.read_csv("./分类2 - 窃电用户特征及预测分析/stolen_electric_analysis_2.csv", dtype=str)
    df2 = data.merge(df2, how="inner", left_on="CONS_NO", right_on="CONS_NO")
    df2[["KWH_END", "KWH_BEGIN", "KWH"]] = df2[["KWH_END", "KWH_BEGIN", "KWH"]].astype(float)
    df2["CHK_STATE"] = df2["CHK_STATE"].astype(int)
    result2 = df2.groupby("CONS_NO").apply(lambda x: get_total_twh(x))
    result2["SUM_KWH"] = (result2["TOTAL_END_KWH"] - result2["TOTAL_START_KWH"]).apply(lambda x: round(x, 2))
    result2.to_csv("./分类2 - 窃电用户特征及预测分析/1-2.csv", index=True)
    # 3
    x_train = result2.iloc[:, 2:]
    y_train = df2[["CONS_NO", "CHK_STATE"]].drop_duplicates().set_index("CONS_NO")["CHK_STATE"].reindex(x_train.index)
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, y_train)
    df3 = pd.read_csv("./分类2 - 窃电用户特征及预测分析/stolen_electric_analysis_3.csv", dtype=str)
    df3 = data.merge(df3, how="inner", left_on="CONS_NO", right_on="CONS_NO")
    df3[["KWH_END", "KWH_BEGIN", "KWH"]] = df3[["KWH_END", "KWH_BEGIN", "KWH"]].astype(float)
    result3 = df3.groupby("CONS_NO").apply(lambda x: get_total_twh(x))
    result3["SUM_KWH"] = (result3["TOTAL_END_KWH"] - result3["TOTAL_START_KWH"]).apply(lambda x: round(x, 2))
    x_test = result3.iloc[:, 2:]
    y_pred = model.predict(x_test)
    pd.DataFrame({
        "CHK_STATE": y_pred
    }, index=x_test.index).to_csv("./分类2 - 窃电用户特征及预测分析/1-3.csv", index=True, index_label="CONS_NO")
