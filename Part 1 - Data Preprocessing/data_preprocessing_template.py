#!/usr/bin/env python
# coding: utf-8

# This piece of code will always be there at the first before any machine learning program. If missing data, encoding of
# categorised data is required then that must be used, templates are there for reference of syntax.
# Some things of course need to be changed

# Data Preprocessing:-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Separating into dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''
# Feature Scaling:-
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''