# Hierarchical clustering
'''
In this problem as we have done in kmeans, we cluster our customers into different categories
based on the features provided 
'''


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

# Dendrogram
'''
To find the optimal number of clusters
scipy contains tools that are optimal for HC and to "plot dendograms"
'''
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
'''
dendogram() takes in the "linkage" matrix so the linkage method
linkage performs agglomerative clustering
"ward" method minimises the variance in each clustering
in kmeans we had to minimise the wcss and here we minimise the wcv(within cluster variance)
'''
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
'''
From the plot we can see that the largest distance can be the right-most blue lines
from just a bit below 250 to a bit above 100. Thats the longest distance
So our threshold line can be anywhere will just have to cut the longest line
So optimal number of clusters can be 5(3 greens and 1 blue)
'''

# Fitting the HC algorithm to our dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

# Visualising the HC Clusters
'''
Plotting only the specific portions where y_kmeans = 0, 1, 2, 3, 4
s -> size
c -> color
'''
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of the dataset')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()








