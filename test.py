#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :test.py
# @Time        :2020/7/20  16:43
# @Author      :Shuhao Chen
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.fftpack import fft
from IPython.display import display

import pywt
import scipy.stats
filename = 'D:/pycharm/pythonProject/ECGclassify/ECGData/ECGData.mat'
ecg_data = sio.loadmat(filename)
print(ecg_data['ECGData'][0][0][0][0][0])
ecg_signals = ecg_data['ECGData'][0][0][0]
ecg_labels_ = ecg_data['ECGData'][0][0][1]
print(type(ecg_data))
print(type(ecg_signals))
print(type(ecg_labels_))
print('数组形状',ecg_labels_.shape)
print('数组的维度数目',ecg_labels_.ndim)
print(ecg_labels_[0][0])
ecg_labels = list(map(lambda x: x[0][0], ecg_labels_))
print(ecg_labels[0])