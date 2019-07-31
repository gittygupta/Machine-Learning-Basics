# Regression template:-

# Data Preprocessing:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 3].values

'''# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
'''

'''# Feature Scaling:-
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''

# Fitting the Regression Model to the dataset
# 


# Predicting new result with a new input
y_pred = regressor.predict([[6.5]])

# Visualising the Regression results

y_pred = regressor.predict(x)
plt.scatter(x, y, color = 'orange')
plt.plot(x, y_pred)
plt.title('Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

y_pred = regressor.predict(x_grid)
plt.scatter(x, y, color = 'orange')
plt.plot(x_grid, y_pred)
plt.title('Regression plot')
plt.xlabel('Position level')
plt.xlabel('Salary')
plt.show()
