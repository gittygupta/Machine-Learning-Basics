# ANN

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Dummy encoding of categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_gender = LabelEncoder()
m = pd.get_dummies(pd.Series(x[:, 1]))
data = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Exited'], axis = 1)
data = pd.concat([m, data], axis=1)
x = data.iloc[:, :].values

x[:, 4] = labelencoder_gender.fit_transform(x[:, 4])
x = x.astype(float)
x = x[:, 1:]

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Feature Scaling:-
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Part 2 - Creating ANN!

# Importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the first input layer and hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
'''Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")
'''

# Adding the 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
'''Dense(activation="relu", units=6, kernel_initializer="uniform")
'''

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making predictions and evaluating the model

# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)