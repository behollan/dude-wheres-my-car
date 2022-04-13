# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:33:27 2022

Strips training and truth data down to a more reasonable size.
Inputs and outputs are assumed to be in the same directory this script is.

@author: bholland
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

num_rows = 200000 # Strips out the first num_rows from the data set

trainData = pd.read_csv("trainData.csv", nrows=num_rows, header = None)
trainTruth = pd.read_csv("trainTruth.csv", nrows=num_rows, header = None)
testData = pd.read_csv("testData.csv", nrows=num_rows, header = None)



min_max_scaler = preprocessing.MinMaxScaler()
trainData_minmax = pd.DataFrame(min_max_scaler.fit_transform(trainData.T)).T

# trainData1, testData1 = train_test_split(trainData_minmax)

trainData.to_csv("trainData_stripped200k.csv", header = False, index = False)
trainTruth.to_csv("trainTruth_stripped200k.csv", header = False, index = False)
testData.to_csv("testData_stripped200k.csv", header = False, index = False)

trainData_minmax.to_csv("trainData_Norm200k.csv", header = False, index = False)