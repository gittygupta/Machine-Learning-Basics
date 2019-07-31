# Thompson sampling

'''
In random_selection we got 1200 approx clicks and in UCB we got 2200 approx clicks
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import math
import random
N = 10000
d = 10
total_reward = 0
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    elif reward == 0:
        number_of_rewards_0[ad] += 1
    total_reward += reward
    
'''
Thompson sampling gave us a reward of 2500 approx, better than UCB and more than double of 
random_selection.
We will get different results each time because of some random betavariate factors. But we will get more
or less similar results.
'''

# Visualising the results
'''
Here also ad 4 is shown the most.
The algo also found this ad to be clicked the most faster as we can see it was shown >9000 times
and much higher times than UCB.
'''
plt.hist(ads_selected)
plt.title('Histogram of ads shown (not clicked)')
plt.xlabel('Ad number')
plt.ylabel('Number of times it was shown')
plt.show()