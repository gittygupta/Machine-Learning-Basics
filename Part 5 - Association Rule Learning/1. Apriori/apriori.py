# Apriori

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
m = []
for i in range(0, 7501):
    for j in range(0, 20):
        m.append(str(dataset.values[i, j]))
    transactions.append(m)
    m = []

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
results = list(rules)
