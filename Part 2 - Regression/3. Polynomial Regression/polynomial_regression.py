# Data Preprocessing:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear Regression model:
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 2)
x_poly = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

# Visualising the Linear Regression results
y_pred = linear_regressor.predict(x)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Linear Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

polynomial_regressor = PolynomialFeatures(degree = 3)
x_poly = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

polynomial_regressor = PolynomialFeatures(degree = 4)
x_poly = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
x_poly = polynomial_regressor.fit_transform(x_grid)
y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x_grid, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

linear_regressor.predict([[6.5]])
linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]]))

