# -*- coding: utf-8 -*-
"""
Plots the first 3 raw traces for each label type

@author: bholland
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = "C:\\repos\dude-wheres-my-car\\data\\"
image_dir ="C:\\repos\dude-wheres-my-car\\images\\first3traces\\"
num_rows = 150

trainData = pd.read_csv(data_dir + "trainData.csv", nrows=num_rows, dtype=np.int32, header = None)
trainTruth = pd.read_csv(data_dir + "trainTruth.csv", nrows=num_rows, dtype=np.int32, header = None)

trainData.insert(0,'label',trainTruth)
for i in range(20):
    matchedData = trainData[trainData['label'] == i].drop('label', axis = 1)
    fig = plt.figure(dpi=160)
    axs = plt.axes()
    matchedData.iloc[range(3)].transpose().plot(ax = axs, title="Label "+ str(i))
    axs.set_xlabel('bin')
    axs.set_ylabel('count')
    axs.set_ylim(bottom = 0, top = 1700)
    axs.legend(title='Trace number')
    axs.grid(visible=True)
    
    fig.savefig(image_dir+"label"+str(i)+".svg", format='svg')
    fig.savefig(image_dir+"label"+str(i)+".jpg", format='jpg')
