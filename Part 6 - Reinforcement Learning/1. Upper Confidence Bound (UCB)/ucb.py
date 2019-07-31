# Upper Confidence Bound
'''
UCB doesnt' actually have it's own predefined class/function. We gotta do it from scratch
'''

'''
We are given 10 ads and we gotta find which ad is technically the best
Here we have 10000 rounds in the dataset which is 10 armed bandit problem i.e, has 10 ads
Each time a user logs in we show him 1 out of the 10 ads and check rewards. The showing of the ads 
will not be random but it'll be based on what the values of confidence bound and average rewards were
for each ad in the previous round.
According to the algorithm we build a confidence bound and a common starting level using the first few 
datapoints. And then we execute the next operation and calculate the next confidence bound and so on.
We continue the steps and for each round the ad with the highest confidence bound is executed and 
shown and based on whether the user clicks on the ad reward is calculated and then the average reward
and the new confidence level is computed and based on this new data formed in the new round, next
steps are taken to decide which ad to be shown next.

Actually this algorithm doesn't require anything known as a dataset. It makes decisions
instantaneously. Then why do we have a dataset?
That's because we certainly cannot do web scraping now or post an actual ad. We just have a set of, 
say, some recorded data based on clicks made and we will make our algorithm work on that. The dataset
we have is not like a "dataset" that we thought until now. It actually is like a stored info just
because we cannot post an actual ad. Its just that we know who will click on which ad and who won't.
It's just an assumption. So let's not get confused.

Thus while we run the algo the algorithm will get to know about the rewards and punishments from the
dataset. And make the next step in the next round as to which ad is to be shown next, just as the 
algo works.
'''


'''
In "random_selection.py" file everytime we assign a random ad to be shown in each ad. Showing of ads
won't depend on anything, not even on any observations of previous round. It'll be pretty random.
And total_reward is summed up. On an average we see 1200 people click the ad out of 10000.
We can see which ad is randomly selected for which user.
Let's see if the number can be increased using UCB.
'''

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
'''
"[0] * d" creates a vector of size "d" with each element being 0
We calculate the sum_of_rewards and number_of_selections and from them we calculate avg reward
and the confidence bound for each particular round. So we have the 10000 wala loop bahar.
When we get the max upper bound we keep track of the ad that gets the max upper bound, in variable 'ad'

We actually allow each ad to get showed at least once before we start running the algo rigorously
Initially number of times an ad is showed is 0 for all. So we allow all the loop to run until all 
the ads are showed at least once, and once it is showed we assign UCB as 1e400 (high value) to it.
But the max_upper_bound becomes 1e400 because that's equal for all and it gets set for ad "0" in the
first iteration of the loop. Thus the 1st round is our trial round

According to algo we show only that ad that has the max upper bound. So "numbers_of_selections += 1"
is done only for that particular ad and not for all the ads as all the ads aren't shown in 1 login

In the first whole iteration of i-loop i.e, in round 0, the 0th ad gets showed obviously. 
In the second round i.e, n = 1, the numbers_of_selection of ad 0 is 1, and as this ad gets shown, the
new computed upper bound decreases. For ad 1, the number_of_selection still remains 0 as it wasn't 
shown in the last round, so it gets upper bound = 1e400 again and thus becomes the ad with highest
upper bound so it gets shown. Similarly all the 10 ads (0 - 9) gets shown in the 1st 10 rounds. Then
the else statement stops executing i.e, none of the ads are assigned upper bound = 1e400 anymore 
'''

import math
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selection[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    '''
    After showing the ad we see if the user has clicked on it or not. We find that from the dataset
    '''
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    '''
    from the ads_selected list we can see that in the last rounds ad 4 is shown more than others
    so it is basically the best ad
    '''
    
# Visualising the results (Histogram)
plt.hist(ads_selected)
plt.title('Histogram of the ads selected')
plt.xlabel('Ad number')
plt.ylabel('Number of times ad shown')
plt.show()
'''
Clearly ad 4 was selected to be shown to the customer the highest number of times
'''

