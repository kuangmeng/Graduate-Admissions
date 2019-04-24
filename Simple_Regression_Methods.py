#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:24:14 2019

@author: kuangmeng
"""

from Read_Dataset import ReadDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from Evaluation import R2_score


x_train, x_test, y_train, y_test = ReadDataset("./data.csv")

# Linear Regression
lr = LinearRegression()    
lr.fit(x_train, y_train)
y_head_lr = lr.predict(x_test)

# Random Forest Regression
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train, y_train)
y_head_rfr = rfr.predict(x_test) 

# Decision Tree Regression
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train, y_train)
y_head_dtr = dtr.predict(x_test) 

# Ridge
rg = linear_model.Ridge(alpha=.5)
rg.fit(x_train, y_train)
y_head_rg = rg.predict(x_test)

# Lasso
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
y_head_lasso = lasso.predict(x_test)

# Bayesion Ridge
by = linear_model.BayesianRidge()
by.fit(x_train, y_train)
y_head_by = by.predict(x_test)

# Neural Network
nn = MLPRegressor(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
nn.fit(x_train, y_train)
y_head_nn = nn.predict(x_test)




# Please use R2_Score to evaluate these Regression methods
if __name__ == "__main__":
    print(R2_score(y_test, y_head_lr))
    print(R2_score(y_test, y_head_rfr))
    print(R2_score(y_test, y_head_dtr))
    print(R2_score(y_test, y_head_rg))
    print(R2_score(y_test, y_head_lasso))
    print(R2_score(y_test, y_head_by))
    print(R2_score(y_test, y_head_nn))


