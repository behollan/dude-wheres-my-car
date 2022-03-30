# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:33:27 2022

Strips training and truth data down to a more reasonable size.
Inputs and outputs are assumed to be in the same directory this script is.

@author: bholland
"""
import pandas as pd

num_rows = 100 # Strips out the first num_rows from the data set

trainData = pd.read_csv("trainData.csv", nrows=num_rows, header = None)
trainTruth = pd.read_csv("trainTruth.csv", nrows=num_rows, header = None)

trainData.to_csv("trainData_stripped.csv")
trainTruth.to_csv("trainTruth_stripped.csv")