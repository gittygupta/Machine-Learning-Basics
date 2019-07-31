#!/usr/bin/env python
# coding: utf-8

# In[43]:


# In the 'Data.csv' folder there is the name of country, age, salary
# What we wanna predict is whether the person has purchased the product or not
# Thus, Country, Age, Salary are INDEPENDENT variabes and Purchased is the DEPENDENT variable


# In[ ]:


# In Jupyter Notebook in between brackets of a function press (shift+tab) to know about the function and it parameters


# In[44]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[45]:


# Importing the datasets
# Dataset must be in the same folder or specify the directory of the csv file
dataset = pd.read_csv('Data.csv')


# In[46]:


dataset         # This Line displays the dataset imported


# In[47]:


# we will create the matrix of the 3 independent variables, Country, Age, Salary

x = dataset.iloc[:, :-1].values
# [x:y, p:q] means we take rows numbered from x to y and columns numbered from p to q
# [:, :-1] means we take all the rows and all the columns except the last one
# .values means we take all the values in the specified region


# In[48]:


# Thus we get the 3 independent variables
x        # its in the form of an m*n array


# In[49]:


# Now we create the matrix/array of dependent variables

y = dataset.iloc[:, -1:].values
# Now we have the dependent and the independent variabes separated 
# if we remove '.values', the data is stored in the variable as type pandas and not type array


# In[50]:


y


# In[51]:


# Missing Data:-
# Happens a lot of time where certain data is missing, like in this case salary of germany and age of spain
# Removing missing observations might be problematic
# So here we tackle that
# What we do is, we fill the missing data with the mean of the rest of the data available
# We use the scikitlearn library to import Imputer which allows to take care of the missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# We create an imputer object first
# The missing values are called 'nan' and strategy to tackle is to find 'mean'
# We can even take the 'median' strategy or the 'most frequent' strategy or many other strategies
# axis = 0 means we are to tackle with the given strategy along the columns
# axis = 1 means along the rows with the same strategy
# By default, (missing_values = 'Nan', strategy = 'mean', axis = 0)

imputer = imputer.fit(x[:, 1:3])
# it fits the tackled data(which is the mean of corresponding columnn in this case) only to the specified columns and not all
# Thus we fit the data using the mean of all the rows and particularly for the given columns

# Now we relace the missing data of the matrix 'x' by the mean of the column found
x[:, 1:3] = imputer.transform(x[:, 1:3])        # We give in the columns need to be transformed


# In[52]:


# Thus we see that the missing data is replaced by the respective means in 'spain', and 'germany'
x


# In[53]:


# Encoding Categorical Data:-
# Country, Purchased are the categorical data because they have categories
# Country has 3 categories: France, Spain, Germany and Purchased has 2 categories: Yes, No
# For the predictions via mathematical equations we gotta encode the string data as numbers

from sklearn.preprocessing import LabelEncoder

# We first create a LabelEncoder object
labelencoder_country = LabelEncoder()
x[:, 0] = labelencoder_country.fit_transform(x[:, 0])        # We import only the country column


# In[54]:


# Thus we change the countries' name to be encoded as integers
# Thus France->0, Germany->1, Spain->2
x


# In[55]:


# A problem that can occur here is, there comes a heirarchy order
# Germany becomes above France due to higher number and Spain is above both which must not happen
# To prevent that we'll use dummy encoding


# In[56]:


''' DUMMY ENCODING:-
    
    In this method, we split the country column into 3 different categories, each for a country
    Example:-
    
    Country        France-code        Spain-code        Germany-code
    France              1                 0                  0
    Spain               0                 1                  0
    Germany             0                 0                  1
    
    Thus here we make sure all of them have values 1 and no heirarchy is being set, and the equations work properly
'''
# To do this we have to use another class under sklearn


# In[63]:


'''
The method of dummy encoding is shown below which is done using pandas and not OneHotEncoder
Cz i couldnt figure out how to do it.  And OHE seems to be deprecated
OHE was printing some shit
'''


# In[57]:


m = pd.get_dummies(pd.Series(x[:, 0]))        # This line is to make the dummies of a given column for given number of rows
m                                             # And returns a pandas.dataframe type


# In[58]:


data = pd.DataFrame({'Country':x[:,0],'Age':x[:,1], 'Salary':x[:,2]})        # Here we convert array 'x' into pandas dataframe
data


# In[59]:


data = data.drop('Country', axis = 1)        # This line is to remove a particular column frm the dataframe, here 'country'
data                     # We actually dont require this line. We could have not added the 'country' column before itself


# In[60]:


data = pd.concat([m, data], axis=1)        # Here we concatenate the dummy list with the 'data' variable
data


# In[71]:


x = data.iloc[:, :].values        # Here we convert dataframe back to numpy.ndarray
x                                 # Thus we get the desired result


# In[72]:


# For 'purchased' variable there are 2 categories, so it has to be either 0 or 1 only so we use LabelEncoder only


# In[73]:


labelencoder_purchased = LabelEncoder()        # Here we dont need to specify the rows and columns cz we need to do it for all
y = labelencoder_country.fit_transform(y)


# In[74]:


y


# In[75]:


# Splitting the dataset into 'training' set and 'test' set :-
# Machine learning models learn to correlate between the training set and the test set, and give future predictions which
# will be based on a new set different from the training set and the test set
# The machine must be well able to perform correlations between the two sets so that it doesn't learn the correlations by heart
# but is able to understand them. So there must not be too much difference between the test and the training set


# In[77]:


from sklearn.model_selection import train_test_split                 # this is the library we'll use
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Takes in the arrays, test size
# If test_size = 0.5, means 1/2 of the data goes to the test set and 1/2 goes to the training set
# test_size = 0.2 means 20% test set, 80% training set
# We can input either test_size or train_size
# Random_state is a random number used for random sampling and splitting


# In[79]:


x_test


# In[80]:


x_train


# In[81]:


y_test


# In[82]:


y_train


# In[83]:


# The correlation is done between the x_train and y_train
# And we will see if the computer can apply this correlation in the test set and whether it can predict based on training set


# In[84]:


# Feature Scaling :-
'''
From the data we see that the Age is in range of 10's and 100's at max
But Salary ranges in 10000's. Thus both of these variables are not to the same scale. During any operation between the two
the Age variable almost gets neglected. So basically we will need to scale down values of both the variables, Age and Salary
to lie between -1 and +1, so that 1 variable doesn't dominate the other during calculations. This is called feature scaling.

There are 2 types :- Standardisation, and Normalisation. Formulas are given with the image attached in the directory
'''


# In[85]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# When we apply scaling to the training set, we need to fit the sc_x object and then transform the training set
# When we apply scaling to the test set, we don't need to fit the sc_x object but directly transform the test set
# We fit it to x_train first so that the test set gets scaled according to the training set
# Question is do we need to scale the dummy variables because they are already 0 or 1?
# Scaling makes all the variables be in the proper scale, so its actually preferrable
# Even if we don't scale that doesn't matter because that won't break the model. Scaling of dummy actually depends on the set
# Sometimes scaling may also lead to - us not knowing which country is which one. Though here we will scale.
# When dealing with huge data we might want to apply feature scaling to the dependent variables as well


# In[88]:


# Like, remember in stats, we sometimes applied scaling only to the 'y' variable and not the 'x' variable. Scaling is not some-
# thing that's always required


# In[90]:


x_train


# In[91]:


x_test


# In[92]:


# Generally we can copy the whole data_preprocessing_template.py file to all the future programs
# Things we don't always require there are: 
# 1. Missing data handling
# 2. Categorising data
# 3. Feature Scaling
# 
# So our basic template will include only:-
# 1. Getting the dataset
# 2. Importing libraries
# 3. Importing the dataset
# 4. Splitting into training and test sets
#
# So you can see all the different things separated into different files, but the whole explanations will be here in this file


# In[ ]:




