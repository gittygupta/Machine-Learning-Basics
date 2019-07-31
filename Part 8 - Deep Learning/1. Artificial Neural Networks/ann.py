# Artificial Neural Networks
'''
Theanos - Uses both CPU and GPU 
GPU is more effective because it can do parallel computations

Tensorflow - Uses both CPU and GPU

Keras - Library that somewhat integrates both of the above (we'll use this)
'''

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
'''
This step is needed because we have categorical data like 'Geography' and 'Gender'
As usual we could ave used OneHotEncoder but it's becoming obsolete so we wont use that
We can label encode 'gender' because its 0 or 1
Then as usual we remove 1 dummy variable to avoid dummy variable trap
and convert the array to 'float' from 'object'

This is my method of doing it
'''
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
'''
In Deep Learning Feature Scaling is absolutely important
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Part 2 - Creating ANN!

# Importing Keras library and packages
'''
"keras" library uses "tensorflow" as backend
"Sequential" module is used to initialise the neural network
"Dense" module is used to create the layers of the neural network
'''
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
'''
This neural network will be a classifier because we will predict yes/no in the end
'''
classifier = Sequential()

# Adding the first input layer and hidden layer
'''
Step 1 - Randomly initialising each node with weights close to 0
Step 2 - Input the 1st observation of the dataset in the input layer, each feature as input
Step 3 - Applying Activation Function (Rectifier Function) for 1 neuron which will forward
         propagate with each neuron's activation being limited by weights
         (We use rectifier function for the hidden layer and sigmoid for the output layer)
         
"add(layer)" adds a layer to the NN
Here we will add 2 layers - input and hidden
Here we gotta play around with how many number of nodes we want in the hidden layer.
There is no rule of thumbs to choose the optimal number of nodes in the hidden layer.
Here we choose the average of the number of layers in the input and output layer.
Here it's (11+1) / 2 = 6 nodes in the hidden layer

Activation function is called "relu"

"output_dim" parameter here specifies the number of nodes to which the layer previous to it
will converge to i.e, the input layer is of size 11 which converges to hidden layer size 6
"input_dim" parameter is to specify the number of nodes in the previous layer because
firstly, we haven't created any layer before the input layer so the comp can't identify
which layer is converging into the hidden layer. So we specify this parameter only for the 
1st step. We don't have to do it for the consecutive layers.


'''
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
'''
"Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")"
The above line is for new keras Dense class. The coded line might become deprecated
'''


# Adding the 2nd hidden layer
'''
Technically we don't need this layer but we may add as many hidden layers as we want
'''
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
'''
The above line is for new keras Dense class. The coded line might become deprecated
Dense(activation="relu", units=6, kernel_initializer="uniform")
'''

# Adding the output layer
'''
In the output layer we will use sigmoid function
"softmax" is a sigmoid function applied when we have 3 or more categorical outputs
'''
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
'''
"optimizer" is the algorithm we are gonna use to find the optimal weights.
We are gonna use "adam" which is the most commonly used Stochastic gradient descent algo
"loss" variable is the thing/function value the algorithm is gonna minimise. Like in 
Linear regression we minimised the least squares. Here using the sigmoid function, the 
error that comes is a logarithmic value. For binary outcomes loss = 'binary_crossentropy'
and for categorical (more than 2 types of outcomes), loss = 'categorical_crossentropy'

"metrics" is a parameter used to determine on which parameters will the loss be calculated.
It can be accuracy, precision, recall anything or all of them combined
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
'''
In the fit method itself we can input the batch size after which the weights will get updated
(if 1, then Stochastic GD, else Batch GD)
We can also include the number of epochs (1 epoch = iteration over the whole dataset)
As we know more the number of iterations over the whole dataset better will be the training

We can view the accuracy increasing (maybe decreasing sometimes) each time in the kernel
Finally accuracy converged to around 86% (approx). Will be different each time
'''
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making predictions and evaluating the model
'''
We take a look at the probabilities of a person leaving the bank
The 2nd line gives the result in terms of true and false so we can validate the results
using the confusion matrix
'''
# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
