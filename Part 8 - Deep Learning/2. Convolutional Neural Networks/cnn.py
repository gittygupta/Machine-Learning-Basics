# Convolutional Neural Network
'''
The keras library while training understands which image belongs to which category by reading
the folder/file name. So always remember to arrange the dataset in the way it is done here 
i.e, dataset -> training, test set -> cats, dogs -> images

Also data preprocessing has been taken care of manually. Data has been split into training and
thetest sets already.

Also we gotta apply feature scaling. Must for every deep learning algo
'''

# Part 1 - Building the CNN

# Importing the keras libraries and packages
'''
Sequential - To create layers
Convolution2D/Conv2D - To create the convolutional layer
Dense - Add fully connected layers to the CNN
The rest just make sense
'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
'''
Parameters:
filter - Number of feature detectors/ number of feature maps in the convolutional layer
kernel_size - Number of rows in the feature detector matrix
              Number of columns in the feature detector matrix
data_format = "channels_last" means 3 will be in the last
"channels first" means 3 will be first in the input_shape parameter
Format of the image: input_shape = (256, 256, 3) i.e 3 Channels (RGB) and matrix of size
(256x256)

Here we have 32 feature detectors of size 3x3 and Format = (64, 64, 3)
'''
classifier.add(Conv2D(filters = 32, 
                      kernel_size=(3,3), 
                      data_format = 'channels_last',
                      input_shape = (64, 64, 3),
                      activation = 'relu'))

# Step 2 - Max Pooling
'''
We take 2x2 matrix on feature map and find max out of the 4 numbers in that 2x2 matrix
so pool_size = (2, 2)
'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
'''
Why don't we lose any of the data while flattening?
That's cz in the high numbers in the feature map we have extracted the spatial features from
the input image thanks to the feature detectors applied. And in the max pooling step we 
extract these high numbers. And when we put these numbers into a single vector we are still
keeping those high numbers. Thus we keep the spatial structure information in the single 
vector.

We don't flatten the whole image without making convolutions because then each input will
contain pixel details and not the features which are a combination of pixels. A single pixel
of the input image can't decide on the feature. obviously.

And these high numbers that we have represent specific features only and not each pixel 

We don't need to input any parameter here cz keras will understand that it'll have to flatten
the previous layer.
'''
classifier.add(Flatten())

# Step 4 - Full Connection
'''
output_dim = units = 128 (quite large, but not too large)
Line 1 - Fully connected layer
Line 2 - Output layer (sigmoid function and binary outcome. So units = 1)
'''
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))

# Compiling the CNN (Error finding and stuff)
'''
adam - Stochastic GD
As we have only 2 categories so binary_crossentropy
And accuracy metric
'''
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])



# Part 2 - Fitting the CNN into the images
'''
We will do image augmentation that is handled by the keras library to prevent overfits
This is a readymade set of code from keras documentation

What it does is it makes certain transformations on the images that we have like, rotating,
sheering etc... Thus creating a batch of new images from 1 single image from the batch of
many images in the dataset, therefore giving a lot more images to train on. Basically it 
enriches the dataset without technically ading new images to it

From the website we took the flow_from_directory method as we have the dataset in a folder

Thus in the code section here the images are augmented, augmented images are generated and
testing is also done in the test set

The libraries automatically detect the 2 classes from the directories thanks to the way the 
images are arranged
'''
from keras.preprocessing.image import ImageDataGenerator
"""
Augmentation
rescale - assigns each pixel value to be between 0 and 1
"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

"""
We first input the location of the file
batch_size -> Number of images after which the weights will be updated
"""
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
"""
validation_steps -> Number of images in the test set
"""
classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)
"""
ALWAYS KEEP batch_size = 32

While training, variables acc -> accuracy on training set
                          val_acc -> accuracy on the test set

We can increase test set accuracy by adding more Convolutional layers and max pooling layers
to the conv layers and by adding more fully connected layers

.fit() method is used when the whole dataset can be fit into the RAM of the pc and no data
augmentation is done.
.fit_generator() method is used when data augmentation is applied which is done here

applying data augmentation implies that our training data is no longer “static” — 
the data is constantly changing.
Each new batch of data is randomly adjusted according to the parameters supplied to 
ImageDataGenerator.

.fit_generator() trains on training set and validates on the test set. Thus you get to see
the accuracy you get on the test set. You can then save your model if you want. For future
predictions on new datasets you can do the following:
    1. steps = number of images in the test set / batch size, where batch_size = 32 (default)
    2. Use 'y_pred = classifier.predict_generator(test_set, steps)'
"""



# Saving the model
from sklearn.externals import joblib 

"""Save the model as a pickle in a file 
"""
joblib.dump(classifier, 'trained_model.pkl') 

"""Load the model from the file
""" 
classifier_from_joblib = joblib.load('trained_model.pkl')  

"""Use the loaded model to make predictions 
classifier_from_joblib.predict(X_test) 
"""

# For future predictions on a different test set
"""steps = number_of_samples_in_new_test_set / batch_size (32)
Make sure the test_set directory is changed and the same on is not used as above
"""
y_pred = classifier.predict_generator(test_set, steps)











