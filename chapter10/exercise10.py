# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


if __name__ == '__main__':
    # 1
    df1 = pd.read_csv("./分类3 - 油浸式变压器/transformer_diagnosis_1.csv", encoding="gbk", index_col="序号")
    x_train = df1.drop(columns=["STATE"])
    le = LabelEncoder()
    y_train = le.fit_transform(df1["STATE"])
    kf = KFold(n_splits=5, shuffle=True)
    model = RandomForestClassifier(n_estimators=20, max_depth=None, criterion="gini", min_samples_leaf=1, random_state=66, n_jobs=-1)
    scores = []
    for train_idx, val_idx in kf.split(x_train, y_train):
        model.fit(x_train.iloc[train_idx], y_train[train_idx])
        scores.append(model.score(x_train.iloc[val_idx], y_train[val_idx]))
    print(np.mean(scores))
    model.fit(x_train, y_train)
    test_data = pd.read_csv("./分类3 - 油浸式变压器/transformer_diagnosis_2.csv", encoding="gbk", index_col="序号")
    x_test = test_data.drop(columns=["STATE"])
    y_pred = model.predict(x_test)
    test_data["STATE"] = le.inverse_transform(y_pred)
    test_data.to_csv("./分类3 - 油浸式变压器/transformer_diagnosis_result.csv", index=True, index_label="序号", encoding="gbk")
    # 2
    plt.figure(2)
    plt.pie(test_data["STATE"].value_counts(), labels=test_data["STATE"].value_counts().index, autopct="%.2f%%", pctdistance=0.8)
    plt.savefig("./分类3 - 油浸式变压器/transformer_diagnosis.jpg")
    plt.show()