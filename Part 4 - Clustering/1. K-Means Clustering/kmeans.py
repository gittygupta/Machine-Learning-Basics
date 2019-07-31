# K-Means Clustering

'''
What actually happens here is we don't have any dependent variable as we don't know
how the clustering of the data is done. So we take only the last 2 independent variables i.e,
Salary and spending score. And based on this we cluster our data
y_kmeans is the dependent variable containing the clustered data
We can classify the data as - 1. Spends with low salary
                              2. Careful with spendings
                              etc, etc...
As in classifier we had the y variable and 2 independent variables which predicted the 2 classes
In this case we feed the kmeans object the 2 dependent variables which then clusters the data
to give the y_kmeans dependent variable

This is actually what happens in the program. 
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')

# Splitting into dependent and independent variables
x = dataset.iloc[:, [3, 4]].values

# Using the elbow method and graph to find the optimal number of clusters
'''
We use a loop to find wcss for clusters 1 to 10
Another name for wcss is 'inertia' and sklearn has a function that finds out wcss for us
'''
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
'''
Plotting the elbow method graph
'''
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''
In this case we have number of optimal clusters as 5
'''

# Now we fit the dataset with the optimal number of clusters
'''
fit_predict method returns the clustered data i.e, for each datapoint, 
it returns the value of each cluster it belongs to
'''
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters
'''
Plotting only the specific portions where y_kmeans = 0, 1, 2, 3, 4
s -> size
c -> color
'''
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
'''
Plotting the Centroids
'''
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of the dataset')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()




# This is just a fun way of naming the clusters based on the graph. Code is the same as visualising

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of the dataset')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()














