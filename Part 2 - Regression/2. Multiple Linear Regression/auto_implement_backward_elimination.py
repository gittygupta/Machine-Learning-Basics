#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

dummy = pd.get_dummies(pd.Series(x[:, 3]))
data = pd.DataFrame({'R&D Spent':x[:,0],'Administration':x[:,1], 'Marketing Spent':x[:,2]})
data = pd.concat([dummy, data], axis=1)
x = data.iloc[:, :].values
x = x.astype(float) 

# Dummy Variable trap:
x = x[:, 1:]

import statsmodels.formula.api as sm
x = np.append(np.ones((50, 1)).astype(int), x, axis = 1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, axis = 1)
    regressor_OLS.summary()
    return x

SL = 0.05
x_opt = backwardElimination(x, SL)
