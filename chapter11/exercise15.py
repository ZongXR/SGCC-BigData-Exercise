# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def is_zeros(_powers: pd.Series) -> bool:
    """
    判断是否电量总为0\n
    :param _powers: city_electric_data的一行
    :return: 是否总为零
    """
    return np.any(_powers.iloc[0:365] > 0)


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./回归3 - 居民异常用电预测/city_electricity_data.csv", encoding="gbk")
    mask = data.apply(lambda x: is_zeros(x), axis=1)
    data = data.loc[mask]
    data.to_csv("./回归3 - 居民异常用电预测/city_electricity_0.csv", index=False)
    city_electricity_sum = data.T
    city_electricity_sum["日期"] = city_electricity_sum.index
    city_electricity_sum["日用电总量"] = city_electricity_sum.sum(axis=1).apply(lambda x: round(x, 2))
    city_electricity_sum = city_electricity_sum[["日期", "日用电总量"]].reset_index(drop=True).iloc[0:365]
    city_electricity_sum.to_csv("./回归3 - 居民异常用电预测/city_electricity_sum.csv", index=True, index_label="序号")
    # 2
    plt.figure(2)
    plt.plot(range(len(city_electricity_sum)), city_electricity_sum["日用电总量"])
    plt.xticks(range(0, len(city_electricity_sum["日期"]), 30), city_electricity_sum["日期"].iloc[0::30], rotation=60)
    plt.xlabel("date")
    plt.ylabel("electricity")
    plt.title("trend")
    plt.savefig("./回归3 - 居民异常用电预测/city_electricity_trend.jpg")
    plt.show()
    # 3
    df3 = city_electricity_sum.iloc[0:181].copy()
    df3["日期"] = df3.index + 1
    df3["日用电总量"] = df3["日用电总量"] / 1000
    with open("./回归3 - 居民异常用电预测/city_electricity_result.txt", "w") as f:
        f.write(str(pearsonr(df3["日期"], df3["日用电总量"])[0]) + "\n")
        x_train, x_test, y_train, y_test = train_test_split(df3[["日期"]], df3["日用电总量"], test_size=0.3, random_state=2)
        model = LinearRegression(n_jobs=-1)
        model.fit(x_train, y_train)
        f.write(str(mean_squared_error(y_test, model.predict(x_test))) + "\n")
        f.write(str(np.sqrt(mean_squared_error(y_test, model.predict(x_test)))) + "\n")

