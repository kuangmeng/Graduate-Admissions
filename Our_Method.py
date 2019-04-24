#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:13:44 2019

@author: kuangmeng
"""

import numpy as np
from Read_Dataset import ReadDataset
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from Evaluation import PPF, R2_score
from sklearn.linear_model import ElasticNet, LassoCV
import xgboost as xgb
import pandas as pd

x_train, x_test, y_train, y_test = ReadDataset("./data.csv")

#以0.8为标准，将录取分为成功1，失败0
y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01  = [1 if each > 0.8 else 0 for each in y_test]
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)

# Decision Tree
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train_01)
y_head_dt = dt.predict(x_test)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gbc.fit(x_train, y_train_01)
y_head_gbc = gbc.predict(x_test)

# SVM
svm_ = svm.SVC(gamma='scale')
svm_.fit(x_train, y_train_01)
y_head_svm = svm_.predict(x_test)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train_01)
y_head_rf = rf.predict(x_test)

# Neural Network
nn = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
nn.fit(x_train, y_train_01)
y_head_nn = nn.predict(x_test)

result_list = [y_head_dt, y_head_gbc, y_head_svm, y_head_rf, y_head_nn]

def Combination(result_list):
    len_ = len(result_list[0])
    ret_list = []
    for i in range(len_):
        tmp_num = 0
        for list_ in result_list:
            tmp_num += list_[i]
        if tmp_num >= 3:
            ret_list.append(1)
        else:
            ret_list.append(0)
    ret_list = np.array(ret_list)
    return ret_list

classification_result = Combination(result_list)

#LASSO MODEL
clf1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4])
clf1.fit(x_train, y_train)
lasso_preds = np.expm1(clf1.predict(x_test))

#ELASTIC NET
clf2 = ElasticNet(alpha=0.00005, l1_ratio=0.9)
clf2.fit(x_train, y_train)
elas_preds = np.expm1(clf2.predict(x_test))

#XGBOOST
clf3=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0.45,
                 learning_rate=0.07,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=500,
                 reg_alpha=0.45,
                 reg_lambda=0.45,
                 subsample=0.95)
clf3.fit(x_train, y_train)
xgb_preds = np.expm1(clf3.predict(x_test))

regression_result1 = 0.45*lasso_preds + 0.25*xgb_preds+0.30*elas_preds

regression_result2 = 0.5*lasso_preds + 0.2*xgb_preds+0.30*elas_preds

def GetFinal(reg_ret1, reg_ret2, cla_ret):
    return_list = []
    for i in range(len(cla_ret)):
        if int(cla_ret[i]) == 1:
            return_list.append(reg_ret1[i])
        else:
            return_list.append(reg_ret2[i])
    return return_list

final_result = GetFinal(regression_result1, regression_result2, classification_result)

solution = pd.DataFrame({"Percent":final_result}, columns=['Percent'])
solution.to_csv("result.csv", index = False)

print(R2_score(y_test, final_result))
print(PPF(y_test_01, classification_result))
