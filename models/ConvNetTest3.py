# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:19:25 2022

@author: ikramer
"""

# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
import numpy

start_time = time.time()
print(start_time-time.time())

# Configuration options
feature_vector_length = 512
num_classes = 20


data_dir = "C:\\Repos\dude-wheres-my-car\\data\\"
#num_rows = 1000000 # Strips out the first num_rows from the data set

trainData = pd.read_csv(data_dir + "trainData.csv", header = None)
trainTruth = pd.read_csv(data_dir + "trainTruth.csv", header = None)
# trainData = pd.read_csv(data_dir + "trainData.csv", nrows=num_rows, header = None)
# trainTruth = pd.read_csv(data_dir + "trainTruth.csv", nrows=num_rows, header = None)

print("Data Load Time: "+ str(time.time()-start_time))

min_max_scaler = preprocessing.MinMaxScaler()
trainData = pd.DataFrame(min_max_scaler.fit_transform(trainData.T)).T

X_train, X_test, y_train, y_test = train_test_split(trainData, trainTruth, test_size=0.30)

  
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Convert target classes to categorical ones
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Set the input shape
X_train.values.reshape(len(X_train),feature_vector_length,1)
input_shape = (feature_vector_length,1)
print(f'Feature shape: {input_shape}')

print("Data Preprocessing Time: "+ str(time.time()-start_time))

# Create the model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

# Test the model after training
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

print("Model Train and Evaluate Time: "+ str(time.time()-start_time))

# Load Test Data Set
MLPtestData = pd.read_csv("testData.csv", header = None)
min_max_scaler = preprocessing.MinMaxScaler()
testData_Norm = pd.DataFrame(min_max_scaler.fit_transform(MLPtestData.T)).T

# Make Predictions
prediction = model.predict(testData_Norm)
predict_classes = prediction.argmax(axis=-1)

# Save to CSV
pd.DataFrame(predict_classes).to_csv("CNNprediction3M2_50epoch.csv", header = False, index = False)

print("Prediction and Write Time: "+ str(time.time()-start_time))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()