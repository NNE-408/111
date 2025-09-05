# 载入相关数据利用机器学习进行预测
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv('test.csv')  

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=33)


xgb_model = xgb.XGBRegressor(max_depth = 5,
                             learning_rate = 0.1,
                             objective = 'reg:linear',
                             n_jobs = -1)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              eval_metric='logloss',
              verbose=100)




