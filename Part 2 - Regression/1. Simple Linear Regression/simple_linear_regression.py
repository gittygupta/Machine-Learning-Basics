# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fiting the Simple Linear Regression model to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# A scatter plot of the points predicted
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color = 'orange')
plt.title('Predictions made on test set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

y_train_pred = regressor.predict(x_train)       

plt.scatter(x_train, y_train, color = 'orange')
plt.plot(x_train, y_train_pred, color = 'blue')
plt.title('Predictions made on training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

