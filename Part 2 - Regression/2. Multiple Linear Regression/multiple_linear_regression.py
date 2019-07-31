#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Here we have to predict the profit(dependent variable) based on the 4 independent variables(country and the 3 spendings)


# In[21]:


# In this case the regression line will be y = a0 + a1x1 + a2x2 + a3x3. Each 'x' being an independent 'spent' variable
# But what about the 'state' variable? 'state' is a categorical variable. We can't add it to our  regression equation
# Refer copy for the details on 'steps of building regression model' and 'backward elimination process'


# In[22]:


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


# In[23]:


# Dummy Encoding
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
'''

# Using OneHotEncoder requires the splitting of categories into numbers using LabelEncoder
# Using pandas doesnt require that
# OneHotEncoder converts the 'x' variable from 'object' to 'float64' which pandas doesn't do. Remember this.
# While backward elimination there causes a type error with 'object' type and not with 'float64' type.
# We can either use OHE or we can just coinvert the ndarray into 'float64'


# In[24]:


# Here we dummy encode with pandas and not OHE and then we convert the 'x' from 'object' type to 'float' type
dummy = pd.get_dummies(pd.Series(x[:, 3]))        # We need the dummies of column 4
data = pd.DataFrame({'R&D Spent':x[:,0],'Administration':x[:,1], 'Marketing Spent':x[:,2]})
data = pd.concat([dummy, data], axis=1)
x = data.iloc[:, :].values
x = x.astype(float)                    # Converting the ndarray into float from object type


# In[25]:


'''onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
'''


# In[26]:


x.dtype        # Ignore the bizzare results
               # See its 'float64'


# In[27]:


# Avoiding the Dummy Variable Trap
# Here we gotta remove 1 dummy variable

x = x[:, 1:]
# Here, we just removed the 'California' column
# Generally this trap is been taken care of by the libraries itself so we don't actually need to do that
# But just to be sure. Don't do it manually like this from the next time


# In[28]:


# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Data preprocessing done


# In[29]:


y_test


# In[30]:


# Fitting the Multiple Linear Regression to the training set
# Similar way. Importing the library and creating the object

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)     # As in Simple linear regression it now correlates all the variables in x_train wth y_train


# In[31]:


# Predicting the test set
y_pred = regressor.predict(x_test)


# In[32]:


y_pred


# In[33]:


'''Backward elimination'''


# In[34]:


# Thus there is a multiple linear dependence. But Move forward to see how we can make our algorithm predict better


# In[35]:


# We could actually build a better model by removing certain variables or features that are not that statistically significant
# to make the best of predictions. To do that, among the 5 method of building a model, we will use 'backward elimination'
# Which is technically the best

# So we actually find a team of independent variables that predict the best value of the dependent variable


# In[36]:


# Building the optimal model using backward elimination


# In[37]:


# Preparation of the matrix
import statsmodels.formula.api as sm

# We gotta add a column of 1's in the regression model because in the regression equation y = a0 + a1x1 .. anxn
# 'a0' is actually 'a0x0' with x0 = 1
# The 'statsmodels' library doesn't do it itself actually and we gotta manually do it
# The LinearRegression class actually can consider a constant 'a0' while making of the regression line
# But this library doesn't work that way, so we gotta add a column of 1's in the ndarray 'x'

x = np.append(np.ones((50, 1)).astype(int), x, axis = 1)
# np.ones() takes in a tuple of number of rows and columns and creates a matrix of 1's all throughout upon which 'x' gets added
# along the rows as we see axis = 1

x        # See. 1st column contains 50 ones.
         # Ignore the bizarre results caused due to conversion into 'float' but don't worry the data is alrighty


# In[50]:


# Now backward elimination begins:

# 'x_opt' will only contain the matrix of features with highest significance on the dependent variable
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
# We write all the columns individually because the algorithm will be removing them 1 by 1 if required

# 1. selecting of significance level

# 2. Fit the full model with all the independent variables
# In this step, we need to newly fit the model'x_opt' using the library 'statsmodels'. Basically we gotta create a new regressor
# The new class is called Ordinary Least Squares
regressor_OLS = sm.OLS(y, x_opt).fit()
# On seeing the list of parameters we see, it takes in the dependent variable first
# and then the optimised matrix of features
# It also says that the intercept is not included so that's why we add the 1's
# The OLS class works on 'float64' type and not on 'object' type. So we had to convert the ndarray


# In[51]:


# 3. Consider the predictor with the highest p-value
# This function returns a table of information
# The lower the p-value of a variable more significant will it be for the model
regressor_OLS.summary()

# Here we have the x0 as 'constant'
# x1, x2-> dummy variables
# x3->R%D Spent
# x4->Admin spent
# x5->Marketing spent
# And we have the corresponding coefficients, p-values etc.
# Significance level is taken as 0.05 by the library itself as it is the most commonly used


# In[52]:


# 4. We gotta remove the variable with p-value > SL. So we see x2 has p-value 0.99 > 0.05 So we gotta remove it
x_opt = x[:, [0, 1, 3, 4, 5]]        # Thus we've removed x2
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()


# In[53]:


x_opt = x[:, [0, 3, 4, 5]]        # Removed x1 corresponding to the 'x_opt' because it has highest p-value = 0.94 > 0.05
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()


# In[54]:


x_opt = x[:, [0, 3, 5]]        # Removed x2 corresponding to the previous 'x_opt'
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()

# p-value can actually not be '0'. Here it's shown 0 cz the value is way too small


# In[55]:


# Thus we see that 'x1' here i.e, the R&D spent has a very high significance towards predicting the dependent variable

# But we also gotta remove 'x2' from above, cz its p-value = 0.06 > 0.05
x_opt = x[:, [0, 3]]        # x2 removed i.e, the 5th column
regressor_OLS = sm.OLS(y, x_opt).fit()
regressor_OLS.summary()


# In[56]:


# Thus for now we only have 1 strong predictor with a very low p-value, i.e, with a very high significance
# Thus the optimal team consists of actually only 1 variable for this dataset.
# Now we will make predictions based on the optimal set we made i.e, the 'x_opt' array

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# In[57]:


y_pred
# So these are the predictions made by the optimal features which have a really high significance towards predicting 'y_pred'


# In[58]:


y_test


# In[ ]:




