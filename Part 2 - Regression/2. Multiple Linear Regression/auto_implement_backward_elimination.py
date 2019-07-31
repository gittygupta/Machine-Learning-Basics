#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[2]:


dummy = pd.get_dummies(pd.Series(x[:, 3]))
data = pd.DataFrame({'R&D Spent':x[:,0],'Administration':x[:,1], 'Marketing Spent':x[:,2]})
data = pd.concat([dummy, data], axis=1)
x = data.iloc[:, :].values
x = x.astype(float) 


# In[3]:


# Dummy Variable trap:
x = x[:, 1:]


# In[4]:


# Here we dont have to do the eliminations manually. We write a function for it which takes in the ndarray and the SL
import statsmodels.formula.api as sm
x = np.append(np.ones((50, 1)).astype(int), x, axis = 1)         # Adding the column of 1's
def backwardElimination(x, sl):
    numVars = len(x[0])                                          # Just to get the number of columns. '0' is the row number
    for i in range(0, numVars):                                  # Checks for the p-value of all the columns individually
        regressor_OLS = sm.OLS(y, x).fit()                       # Fits into the regressor
        maxVar = max(regressor_OLS.pvalues).astype(float)        # Finds the maximum p-value among them as type 'float'
        if maxVar > sl:                                          # Checks if that 'max p-value' is >SL
            for j in range(0, numVars - i):                   # Then in this loop we just check which index has the max p-value
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, axis = 1)               # Then we just remove that column from the array along axis = 1
    regressor_OLS.summary()                                     # Then we generate the summary. Actually not an imp step
    return x

# regressor_OLS.pvalues returns an array of the p-values, which can be accessed based on each index
# This is just a loop that does all the function by itslef and we don't have to do these things manually


# In[5]:


SL = 0.05
x_opt = backwardElimination(x, SL)


# In[6]:


x_opt           # Consists only of the rows of 1's and the most significant variable i.e, R%D Spent


# In[ ]:




