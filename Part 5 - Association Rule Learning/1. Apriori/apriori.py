# Apriori
'''
Stores use Association Rule learning to predict where to place the products in a store

We use the "apyori.py" file to do everything. This file is readily available on the web
'''

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
'''
We write "header = None" because otherwise python is treating the 1st row of information as the
column headings which we do not want. So by giving this extra parameter we make sure that there are
no column headings and each column is treated as 0, 1, 2, 3... etc and actually there is nothing
known as columns as such in this dataset. We just have a rows of informations of the transactions
made 1 after the other in the store and each row being a list of what that customer actually bought

Here we have a dataset of 7500 transactions made over a week, of same or different customers to 
analyse on.

The APRIORI model expects a list of lists actually
'''
transactions = []
m = []
for i in range(0, 7501):
    for j in range(0, 20):
        m.append(str(dataset.values[i, j]))
    transactions.append(m)
    m = []
'''
"m" is the list for 1 customer which is appended to "transactions" list becoming a list of lists
and not a pandas dataframe object
'''

# Training Apriori on the dataset
'''
We will use apyori file instead of using open source functions
apriori takes in the list as input and gives the rules as output
**kwargs incude minimum support, confidence, lift and, max and min length of relation

for minimum support we must choose a number of items that are bought rather frequently
It depends from dataset to dataset. You gotta choose it on your own, tho we can change it anytime
Even minimum confidence and lift depends on our satisfaction of what rules the function returns
until we get the strongest rule according to us

In this case we took,
min_support as a product being bought 3 times a day, so 3*7 times a week
Our dataset consists of the transactions being done over a week
so min_support = 3*7/7500 = 0.003 i.e, 0.3%
We gotta shuffle around with the min_confidence
Too high confidence might not give any rules at all because no rule may not be that highly true 
for all the data in the dataset. Here we take it as 20%
min_lift of 3 might be good  

We gotta spend some time shuffling around the parameters to find out their optimal values that give
us the right rules to train the model

apriori function returns the rules sorted by their relevance
Sorting is not always done wrt the lift but considering all the 3 factors support, confidence, lift.
'''

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# Visualising the results
'''
results is a variable of all the rules containing the variable support, confidence, lift and also
which product is associated with which product.
The first line creates the list of rules but it doesn't show which product is associated to which one
To see that just print the particular thing in the console like "result[0]" or something like that
It shows everything - support, items associated, confidence and the lift

Now based on these rules the analysts manage which rules to go with and combine them all
to arrange products on a mall and thus gain profits from it
'''
results = list(rules)






























