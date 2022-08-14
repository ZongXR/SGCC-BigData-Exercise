# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    # 1
    data = pd.read_csv("./回归1 - 电网故障报修受最高温度的关系/gridfaults_complaints.csv", encoding="gbk")
    df1 = data[data["最高气温"] >= 25].copy()
    df1.to_csv("./回归1 - 电网故障报修受最高温度的关系/gridfaults_complaints_1.csv", index=False, encoding="gbk")
    # 2
    plt.figure(2)
    plt.scatter(df1["最高气温"], df1["故障报修受理数量"])
    plt.xlabel("°C")
    plt.ylabel("number")
    plt.title("scatter")
    plt.savefig("./回归1 - 电网故障报修受最高温度的关系/gridfaults_complaints.jpg")
    plt.show()
    # 3
    pearsonr_value = pearsonr(df1["最高气温"], df1["故障报修受理数量"])[0]
    x_train, x_test, y_train, y_test = train_test_split(df1[["最高气温"]], df1["故障报修受理数量"], test_size=0.3, random_state=11)
    model = LinearRegression(n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse_value = mean_squared_error(y_test, y_pred)
    mse_root_value = np.sqrt(mse_value)
    predict_value = model.predict([[38]])[0]
    with open("./回归1 - 电网故障报修受最高温度的关系/gridfaults_complaints_result.txt", "w", encoding="gbk") as f:
        f.write(str(round(pearsonr_value, 2)) + "\n")
        f.write(str(round(mse_value, 2)) + "\n")
        f.write(str(round(mse_root_value, 2)) + "\n")
        f.write(str(round(predict_value, 2)) + "\n")



