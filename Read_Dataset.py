# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'

def ReadDataset(file):
    # reading the dataset
    data = pd.read_csv(file, sep = ",")

    data.drop(["Serial No."], axis = 1, inplace = True)

    data=data.rename(columns = {'Chance of Admit ':'Chance of Admit'})
    y = data["Chance of Admit"].values
    x = data.drop(["Chance of Admit"], axis = 1)
    print(y)
    # separating train (80%) and test (%20) sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

    # normalization
    scalerX = MinMaxScaler(feature_range=(0.0, 1.0))
    x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
    x_test[x_test.columns] = scalerX.fit_transform(x_test[x_test.columns])
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = ReadDataset("./data.csv")