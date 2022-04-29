# -*- coding: utf-8 -*-
"""
Training, testing, and evaluation pipeline for AI/ML Challenge 2022

This script configures a machine learning model backend, loads the training 
dataset (including pre-processing), trains the model, evaluates performance,
and (optionally) performs a prediction run on the challenge dataset. 

@author: Dude Where's my Car
"""

# Imports
import pandas as pd
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import time
import matplotlib.pyplot as plt
import logging

# Machine learning model definitions
import Models

start_time = time.time()

'''
Configuration options
'''
debug_level = logging.INFO     # Logging level, DEBUG, INFO, WARNING, etc.
feature_vector_length = 512
num_classes = 20
save_model = True
model_name = "convNet2"

full_run = True # Whether or not to train on the full dataset or a subset
num_rows = 1000000 # Number of row for a partial training run, ignored is full_run is True
test_split = 0.20 # Percent of train_data_file to use for evaluation after fitting

# Training file locations
data_dir = "C:\\Repos\dude-wheres-my-car\\data\\"
train_data_file = data_dir + "trainData.csv"
train_truth_file = data_dir + "trainTruth.csv"

# Evaluation options
eval_challenge_data = False # Whether or not to evaluate the challenge dataset
eval_data_file = data_dir + "testData.csv"
eval_results_file = data_dir + "results\\MLP_FullTrain_0420.csv"

# Model options
validation_split = 0.30 # Percent of test data to use for the validation split in training 
batch_size = 128
epochs = 50
model = Models.ConvNet2() # Select model from Models.py file definition

model_params = {
    'loss':'categorical_crossentropy', 
    'optimizer':'adam',
    'metrics': ['accuracy']
    }
fit_params = {
    'verbose': 1,
    'callbacks': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    }

eval_params = {
    'verbose': 1
    }


'''
Setup the logger
'''

logging.basicConfig(level=debug_level, format='[%(asctime)s] :: %(levelname)s :: %(message)s')
logging.debug('Debug level set to: %s', str(debug_level))

'''
Load the dataset
'''
load_params = {'header': None}
if not full_run:
    logging.info("Partial run selected...")
    load_params = load_params | {'nrows': num_rows}
    
logging.info("Loading datafile: %s", train_data_file)
train_data = pd.read_csv(train_data_file, **load_params)

logging.info("Loading truthfile: %s", train_truth_file)
train_truth = pd.read_csv(train_truth_file, **load_params)


logging.info("Data Load Time: "+ str(time.time()-start_time))


'''
Pre-process the dataset
1. Row normalization (0 to 1)
2. Test/validation datas split
3. Recast inputs as float 32
4. Convert labels to keras categoricals
5. Set the input shape
'''
temp_time = time.time()
logging.info("Starting pre-processing")
min_max_scaler = preprocessing.MinMaxScaler()
train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data.T)).T

logging.info("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(train_data, train_truth, test_size= test_split)

  
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Convert target classes to categorical ones
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Cast input to appropriate shape
X_train.values.reshape(len(X_train),feature_vector_length,1)
logging.info(f'Feature shape: ({feature_vector_length},1)')
logging.info("Data Preprocessing Time: "+ str(time.time()-temp_time))

'''
Setup the model
1. Configure and compile the model
'''
# Compile the model
logging.info("Compiling model")
model.compile(**model_params)

'''
Fit the model
'''
logging.info("Start model fitting")
temp_time = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, **fit_params)
logging.info("Fit complete")
logging.info("Model Train Time: "+ str(time.time()-temp_time))

# Test the model after training
logging.info("Evaluate model on test split")
temp_time = time.time()
test_results = model.evaluate(X_test, y_test, **eval_params)
logging.info(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
logging.info("Model Evaluate Time: "+ str(time.time()-temp_time))

'''
Save the model
'''
if save_model:
    model.save(data_dir + "\\saved_models\\" + model_name)
    
'''
Evaluate the challenge test set data
1. Load the challenge dataset
2. Preprocess the dataset
3. Make a prediction run 
4. Save the results for submission
'''
# Load Test Data Set
if eval_challenge_data:
    logging.info("Evaluating the 2022 AI/ML Challenge datset.")
    temp_time = time.time()
    logging.info("Loading challenge dataset.")
    eval_data = pd.read_csv(eval_data_file, header = None)
    logging.info("Normalizing challenge dataset.")
    min_max_scaler = preprocessing.MinMaxScaler()
    eval_data_norm = pd.DataFrame(min_max_scaler.fit_transform(eval_data.T)).T
    
    # Make Predictions
    logging.info("Classifying challenge dataset.")
    prediction = model.predict(eval_data_norm)
    predict_classes = prediction.argmax(axis=-1) # Take the most probable class
    
    # Save to CSV
    logging.info("Saving predictions to "+eval_results_file)
    pd.DataFrame(predict_classes).to_csv(eval_results_file, header = False, index = False)
    
    logging.info("Prediction and Write Time: "+ str(time.time()-temp_time))

'''
Plot some metrics for accuracy and loss
'''
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
    
''' 
Confusion Matrix
'''
test_predict =  model.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=-1), test_predict.argmax(axis=-1),normalize='true')
fig = plt.figure(dpi=160, figsize=(10,10))
axs = plt.axes()
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap=plt.cm.Blues, ax = axs, values_format='.2f')
plt.show()