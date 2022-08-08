#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train_feature.csv')
data.head(20)

da1 = data.groupby('日期').mean()[['辐照度','风速']]
da1.columns = ['辐照度均值','风速均值']

# （二）统通过学习一段时间内的环境数据和对应的实际太阳辐指数，训练模型，通过给定某日期预测的环境数据，预测当天的实际太阳辐射指数，并将预测结果保存至predict_result.csv，形式见下表，其中程序保存“国能日新太阳辐射指数预测2.py”。


train_label = pd.read_csv('train_label.csv')
train_label = train_label.set_index('日期')

da2_train = data[['日期','辐照度', '风速', '风向', '温度', '湿度', '气压']].groupby('日期').agg(['mean','std','min','max','median','var','skew'])

test_feature = pd.read_csv('test_feature.csv')
da2_test = test_feature[['日期','辐照度', '风速', '风向', '温度', '湿度', '气压']].groupby('日期').agg(['mean','std','min','max','median','var','skew'])

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor().fit(da2_train, train_label)
y_pre = dt.predict(da2_test)

predict_result = pd.DataFrame(columns = ['id','Ppi'])

predict_result['id'] = test_feature['日期'].unique()
predict_result['Ppi'] = y_pre

predict_result.to_csv('predict_result.csv', encoding = 'gbk', index = False)

#后面为用三种算法建立模型进行的预测结果

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

predict_result = pd.DataFrame(columns = ['id','Ppi0','Ppi1','Ppi2'])
predict_result['id'] = test_feature['日期'].unique()

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in range(len(models)):
    m = models[i].fit(da2_train, train_label)
    y_pre = m.predict(da2_test)
    predict_result['Ppi'+str(i)] = y_pre


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

predict_result = pd.DataFrame(columns = ['id'])
predict_result['id'] = test_feature['日期'].unique()
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
models = [lr,dt,rf]
for model in models:
    m = model.fit(da2_train, train_label)
    y_pre = m.predict(da2_test)
    predict_result['Ppi_'+str(model)[:2]] = y_pre

predict_result.to_csv('predict_result2.csv', index = False)

# 多次运行发现：
# Linear_model很稳定，无变化
# DecisionTreeRegressor部分有变化（10%以内）
# RandomForest变化较大，有的会超过20%
# （三）依据上题所得预测太阳辐射指数与实际太阳辐射指数计算平均绝对误差MAE，MAE的计算公式如下：

result = pd.read_csv('test_answer.csv')

predict_result['id'] = predict_result.index.values +1

result2 = pd.merge(predict_result, result, on ='id')
mae = []
for i in range(1,4):
     mae.append(sum(abs(result2.iloc[:,-1] - predict_result.iloc[:,i ]))/(result.shape[0]))

result2.to_csv('result2.csv', index = False)


from sklearn.metrics import mean_squared_error
MSE  = []
for i in range(1,4):
    MSE.append(mean_squared_error(result.iloc[:, 1], predict_result.iloc[:, i]))
print('MAE: ', mae)
print('MSE: ', MSE)


# In[ ]:




