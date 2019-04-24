#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:24:14 2019

@author: kuangmeng
"""
import numpy as np
from Read_Dataset import ReadDataset
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from Evaluation import PPF


x_train, x_test, y_train, y_test = ReadDataset("./data.csv")

#以0.8为标准，将录取分为成功1，失败0
y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01  = [1 if each > 0.8 else 0 for each in y_test]
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)

# Logistic Regression
lrc = LogisticRegression()
lrc.fit(x_train,y_train_01)
y_head_lrc = lrc.predict(x_test)

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

# SGD
sgd = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 10, tol = 1)
sgd.fit(x_train, y_train_01)
y_head_sgd = sgd.predict(x_test)

# Neural Network
nn = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
nn.fit(x_train, y_train_01)
y_head_nn = nn.predict(x_test)
  

# Please use "Precise, Recall, F1" to evaluate these Classification methods
if __name__ == "__main__":
    print(PPF(y_test_01, y_head_lrc))
    ...
    ...

