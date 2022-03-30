# -*- coding: utf-8 -*-
"""
Plots the first 3 raw traces for each label type, normalized between [0 1]

@author: bholland
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data_dir = "C:\\repos\dude-wheres-my-car\\data\\"
image_dir ="C:\\repos\dude-wheres-my-car\\images\\first3traces_norm\\"
num_rows = 150

trainData = pd.read_csv(data_dir + "trainData.csv", nrows=num_rows, dtype=np.int32, header = None)
trainTruth = pd.read_csv(data_dir + "trainTruth.csv", nrows=num_rows, dtype=np.int32, header = None)

min_max_scaler = preprocessing.MinMaxScaler()
trainData_minmax = pd.DataFrame(min_max_scaler.fit_transform(trainData.T)).T

trainData_minmax.insert(0,'label',trainTruth)
for i in range(20):
    matchedData = trainData_minmax[trainData_minmax['label'] == i].drop('label', axis = 1)
    fig = plt.figure(dpi=160)
    axs = plt.axes()
    matchedData.iloc[range(3)].transpose().plot(ax = axs, title="Label "+ str(i) +" normalized")
    axs.set_xlabel('bin')
    axs.set_ylabel('count')
    axs.set_ylim(bottom = 0, top = 1)
    axs.legend(title='Trace number')
    axs.grid(visible=True)
    
    fig.savefig(image_dir+"label"+str(i)+"_normalized.svg", format='svg')
    fig.savefig(image_dir+"label"+str(i)+"_normalized.jpg", format='jpg')
