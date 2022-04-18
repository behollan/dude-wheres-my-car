# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:28:57 2022

Usage: Add a function that returns a keras model object. Set the "model"
variable in the trainTestEvalPipeline.py to be the output of your function.

@author: ikramer (original), modified by bholland
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout

def ConvNet1(input_shape = (512,1), filters = 32, kernel_size = 5, pool_size = 5, 
                 strides = 2, num_classes = 20):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, strides=strides))
    model.add(Flatten())
    model.add(Dense(filters, activation='relu'))
    model.add(Dense(filters, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def ConvNet2(input_shape = (512,1), filters = 32, kernel_size = 5, pool_size = 2, 
                 strides = 2, num_classes = 20, dropout = 0.5):

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def MLP(input_shape = (512,1), num_classes = 20):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model