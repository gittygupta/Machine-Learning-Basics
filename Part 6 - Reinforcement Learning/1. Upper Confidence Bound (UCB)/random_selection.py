# Random Selection
'''
In "random_selection.py" file everytime we assign a random ad to be shown in each ad. Showing of ads
won't depend on anything, not even on any observations of previous round. It'll be pretty random.
And total_reward is summed up. On an average we see 1200 people click the ad out of 10000.
Let's see if the number can be increased using UCB.

We can see which ad is randomly selected for which user
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()