#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[8]:


dummy = pd.get_dummies(pd.Series(x[:, 3]))        # We need the dummies of column 4
data = pd.DataFrame({'R&D Spent':x[:,0],'Administration':x[:,1], 'Marketing Spent':x[:,2]})
data = pd.concat([dummy, data], axis=1)
x = data.iloc[:, :].values


# In[9]:


x = x.astype(float)


# In[11]:


x


# In[12]:


x.dtype


# In[ ]:




