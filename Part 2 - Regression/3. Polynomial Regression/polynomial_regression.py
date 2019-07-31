# Polynomial Regression:
# Technically not linear regrression
# In this case the salary is increasing exponentially wrt the level. Position is equivalent to level number
# So position is not considered as another significant variable as it is redundant


# Data Preprocessing:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Separating into dependent and independent variables
# We technically don't need the 'Position' column cz it's kinda encoded in terms of the level actually like Business Analyst = 1
# We want matrix of features 'x' to be seen as a matrix and not a vector. So we do it like this.
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# We don't split the dataset because the data is too less. But we will predict by giving in some random values
# Now it's a matrix and not a vector
# It is always better to keep 'x' as a matrix evem if it has only 1 column and 'y' as a vector
# We will build both linear and polynomial regression models to compare them

# Linear Regression model:
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Polynomial Regression model:
# This class is imported from preprocessing library and not the linear_model library
# PolynomialFeatures takes in the degree 
# Thus it creates x^1, x^2 ... x^n different variables based on the degree entered
# degree = 2, by default

from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 2)
x_poly = polynomial_regressor.fit_transform(x)        # We use fit_transform cz it will create new values and transform
                                                         # the empty variable to non-empty
    
# There is a column of 1's for the constant 'b0' of the equation because we can do backward elimination if needed
# Thus we can understand that polynomial regression is somewhat like Multiple linear regression, only with some extra libraries

# Now we create a new multiple linear regression model with the 'x_poly' values this time
# Thus 'linear_regression_2' is our actual polynomial regression model
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

y_pred
# These are not great predictions
# Some even come in -ve. LOL XD

# Visualising the Polynomial Regression results

y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# To predict even better we can even play around with degrees. Here we changed it to 3
polynomial_regressor = PolynomialFeatures(degree = 3)
x_poly = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

# And plot it
y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# For fun lets see what will degrees = 4 reveals

polynomial_regressor = PolynomialFeatures(degree = 4)
x_poly = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

# And plot it
y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# But we see that out model makes good predictions but it's plotting straight lines in between predictions and not smooth curves
# We will tackle that here
# Out model contains level 1-10 incrementing each by 1
# To get a smoother curve we can increment it by 0.1 or 0.01 which will predict y_pred for each of 'x_grid'

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
# We made an array with values between min(x) and max(x) with incrementation of 0.1 at each step
# We had to reshape it just to make sure it is an array with 'len(x)' rows and only 1 column just as it was in 'x'

# And now we plot
# For this we had to transform x_poly
x_poly = polynomial_regressor.fit_transform(x_grid)
y_pred = linear_regressor_2.predict(x_poly)
plt.scatter(x, y, color = 'orange')
plt.plot(x_grid, y_pred)
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# Actually even smoother

# Predicting new result with a new input
# Using Linear Regression Model:
# We can actually use this method using 1 constant, but we have to send the data in the form of a 2-D array only

linear_regressor.predict([[6.5]])

# Using polynomial regression model:

linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]]))
# We put polynomial_regressor object inside because it makes the polynomial object that way with the 1's and 6.5 and then 
# shows us the results
# Now this is a much accurate result
