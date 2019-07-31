# CNN

# Part 1 - Building the CNN

# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, 
                      kernel_size=(3,3), 
                      data_format = 'channels_last',
                      input_shape = (64, 64, 3),
                      activation = 'relu'))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))

# Compiling the CNN (Error finding and stuff)
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN into the images
'''Taken from keras documentation
'''
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

# Saving the model
from sklearn.externals import joblib 

"""Save the model as a pickle in a file 
"""
joblib.dump(classifier, 'filename.pkl') 

"""Load the model from the file
""" 
classifier_from_joblib = joblib.load('filename.pkl')  

"""Use the loaded model to make predictions 
classifier_from_joblib.predict(X_test) 
"""

# For future predictions on a different test set
"""steps = number_of_samples_in_new_test_set / batch_size (32)
Make sure the test_set directory is changed and the same on is not used as above
"""
y_pred = classifier.predict_generator(test_set, steps)