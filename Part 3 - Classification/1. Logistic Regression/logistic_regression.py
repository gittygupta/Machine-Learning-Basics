# Logistic Regression


# Data Preprocessing:-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# 'Purchased' is the dependent variable
# Statement : There is an ad of something on social media and the features of the person are given and 
# we predict whether he will buy the product or not 

# Separating into dependent and independent variables
x = dataset.iloc[:,[2, 3]].values        # Age and estimated salary. User ID, gender don't matter
y = dataset.iloc[:, 4].values           # Buys or not

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# We gotta feature scale this
# Feature Scaling:-
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Fitting Logistic Regression model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the CONFUSION MATRIX
# Confusion matrix gives us a count of the correct and incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Takes in the actual value first, and then the predicted value

'''
In binary predicitions like these from the table of cm we see:
if the row = 0 and column = 0, the number in that particular cell represents the total number of correct preds
Here 65 and 24 are the correct predictions
where, 0 -> people who didn't buy and 65 of those predictions were correct
and 1 -> people who bought and 24 of those predictions were correct
8 predictions were expected to be '0' but were predicted to be '1'
3 predictions were expected to be '1' but were predicted to be '0'
Basically we got 65+24 = 89 predictions correct out of the 100 in the test set
'''

# Visualising the Training set results

'''
To make the area plot we take each pixel point as a social media user and make predictions
Its technically a new observation with a jump of 0.01 in the age and estimated salary
Considering both the 2 features if the clssifier predicts 0 then it colorises the pixel as red otherwise green
And on separating we get a linear classifier
In the meshgrid we do -1 and +1 so that we increase the graph area and the scatter plot doesn't touch ends
and the graph looks clean and linear for the big amount of the data. We do this for both the columns
contourf creates the contour of the color difference
In the loop we plot for all the real points according to the color difference
'''

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''
This plot is a 'LINEAR' CLASSIFICATION plot
In this graph we see the training set on which the model is trained
The scatter plot shows the actual points of y_train
The 2 different areas represent the prediction regions, red-not bought, green-bought
The area plot we see is that of y_pred
Some green points lie on the red region and some red points on the green.
These are the points that don't exactly fit on the linear classification line separating the two categories
'''



# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''
Here we can even count every of the 11 points that are wrongly predicted
The area/regions remain the same as it is the one on which the models have been trained
'''














