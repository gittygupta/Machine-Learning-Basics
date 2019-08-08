#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Dummy Encoding
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
'''
dummy = pd.get_dummies(pd.Series(x[:, 3]))
data = pd.DataFrame({'R&D Spent':x[:,0],'Administration':x[:,1], 'Marketing Spent':x[:,2]})
data = pd.concat([dummy, data], axis=1)
x = data.iloc[:, :].values
x = x.astype(float)

'''onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
'''
x = x[:, 1:]

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set
y_pred = regressor.predict(x_test)

'''Backward elimination'''
import statsmodels.formula.api as sm
x = np.append(np.ones((50, 1)).astype(int), x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
