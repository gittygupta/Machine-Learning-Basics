#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Taking care of missing data:-
# Refer 'data_preprocessing_whole_tutorial.ipynb' file

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

