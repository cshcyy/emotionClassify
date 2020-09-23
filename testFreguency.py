#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :testFreguency.py
# @Time        :2020/9/23  13:44
# @Author      :Shuhao Chen
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.fftpack import fft
from IPython.display import display
from sklearn import svm
import pywt
import scipy.stats
from feature_selector import FeatureSelector
import datetime as dt
from collections import defaultdict, Counter

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

filename = 'D:/pycharm/pythonProject/ECGclassify/ECGData/data1.mat'
ecg_data = sio.loadmat(filename)
ecg_signals = ecg_data['data1'][0][0][0]
ecg_labels_ = ecg_data['data1'][0][0][1]
ecg_labels = ecg_labels_


list_labels = ecg_labels
list_features = []

# for signal in ecg_signals:
# 	features = []
# 	features += get_features(signal)
# 	list_features.append(features)

Fs = 5
N = len(ecg_signals[0])
print(N)
x = range(N)
plt.plot(x,ecg_signals[0])
y = ecg_signals[0]
plt.show()

ff = np.arange(0,Fs,Fs/N)
print(len(ff))
x= ff
fft_y=fft(y)
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)              #取复数的角度

plt.figure()
plt.plot(x,abs_y)
plt.title('双边振幅谱（未归一化）')

plt.figure()
plt.plot(x,angle_y)
plt.title('双边相位谱（未归一化）')
plt.show()

normalization_y=abs_y/(N/2)#归一化处理（双边频谱）
print(normalization_y)
normalization_y[0] = normalization_y[0]/2  #直流分量归一化应该除以N
print(normalization_y[0])
plt.figure()
plt.plot(x,normalization_y,'g')
plt.title('双边频谱(归一化)',fontsize=9,color='green')
plt.show()
# np.arange(0,Fs,Fs/N)
half_x = x[range(int(N/2))]                                  #取一半区间
normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
plt.figure()
plt.plot(half_x,normalization_half_y,'b')
plt.title('单边频谱(归一化)',fontsize=9,color='blue')
plt.show()