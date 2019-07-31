#!/usr/bin/env python
# coding: utf-8

# In[43]:


# We make a simple linear regression model that finds out the correlation between the years of experience and salaries
# That would probably help to find out the regression line of the best fit
# We have data given upto 10 years. Suppose someone with 15/20 years of experience comes into the company then the regression
# line will help the machine learning model predict his salary


# In[44]:


# Simple linear regression intuition:-
# For regression line of Y on X, Y is the dependent variable and X is the independent variable
# Example: Y is the salary and X is the years of experience. Y = aX + b


# In[45]:


# Simple Linear Regression:-


# In[46]:


# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, :-1].values        # Independent variable : Years of experience
y = dataset.iloc[:, 1].values          # Dependent variable : Salary

# When we give ':-1' instead of just specifying the column, it is known as the matrix of features
# When we directly specify the column, it is known as the Vector variable, here the dependent variable is the vector
# You can see the noticeable difference while displaying. Matrix displayed data in each line, but vectors in 1 single line

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# In[47]:


x_train


# In[48]:


x_test


# In[49]:


y_train


# In[50]:


y_test


# In[51]:


# The machine will train itself to correlate between the x_train and y_train and will be tested on x_test to predict y_test
# Generally feature scaling is taken care by the libraries itself. For Simple Linear Regression we don't need feature scaling


# In[52]:


# Fiting the Simple Linear Regression model to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# As usual we create an object of the imported class. Here it is called regressor which will be fitted into the training set
# regressor.fit()takes in 2 parameters. 1st- the independent variable, 2nd- the target/dependent variable which the pc predicts
# On execution the model will already learn the correlation between the x_train and y_train
# Obviously we don't need to transform anything only fit so we use only the 'fit' method only


# In[53]:


# Now the simple linear regression model is the machine which learnt on the training set i.e, the x_train and y_train.
# It learnt the correlation between the two training sets


# In[54]:


# Predicting the test set results:-
# The model didn't get trained by the x_test. So we will store the predictions that the machine learning model is going to make
# in a 'vector' variable 'y_pred'


# In[55]:


y_pred = regressor.predict(x_test)        # The model(regressor) is going to make the predictions based on the x_test set


# In[56]:


y_pred        # These are the predicted values. May or may not be the same as y_test


# In[57]:


y_test


# In[58]:


# 'y_test' is the vector variable of the real salaries and 'y_pred' is the vector variable of the salaries that the people with
# work experiences in the 'x_test' should have actually had, based on the training of the model by the x_train and y_train sets


# In[59]:


# Visualising the differences
# Here we can see that what is actually the difference between what is predicted and what the actual value is.


# In[60]:


# A scatter plot of the points predicted and the points actually there
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color = 'orange')
plt.title('Predictions made on test set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
# Scatter plot shows the actual data and the line shows how the predicted data goes


# In[61]:


y_train_pred = regressor.predict(x_train)       # Predictions on the training set
                                                # Seems like the results will come same but it doesn't


# In[62]:


y_train_pred


# In[69]:


plt.scatter(x_train, y_train, color = 'orange')
plt.plot(x_train, y_train_pred, color = 'blue')
plt.title('Predictions made on training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
# Scatter plot shows the actual data and the line shows how the predicted data goes


# In[ ]:


# technically the regression model is the line of best fit
# all the points on the scatter plot are the points on which the simple linear regression model was trained
# as the training was done with the training set i.e, x_train and y_train, the x_test vs y_pred line is the same regression
# line with the same slope and intercept as that of x_train vs y_train, obviously. But with different set of points
# So while plotting the test set graph we may or may not change the variables of plotting of the line, doesn't matter at all

# Remember the 'machine' is the simple linear regression model and the 'learning' is the learning from the training set i.e,
# both the x_train and y_train

