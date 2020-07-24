#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :Network.py
# @Time        :2020/7/24  14:42
# @Author      :Shuhao Chen
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle, column_or_1d


# 计算特征的函数集合
# 计算熵值
def calculate_entropy(list_values):
	counter_values = Counter(list_values).most_common()
	probabilities = [elem[1] / len(list_values) for elem in counter_values]
	entropy = scipy.stats.entropy(probabilities)
	return entropy


# 计算各种统计特征
# 方差
# 标准偏差
# 均值
# 中位数
# 第5,25,75,95个百分点
# 均方根值；振幅值平方的平均值的平方
# 导数的均值(还没考虑）
def calculate_statistics(list_values):
	n5 = np.nanpercentile(list_values, 5)
	n25 = np.nanpercentile(list_values, 25)
	n75 = np.nanpercentile(list_values, 75)
	n95 = np.nanpercentile(list_values, 95)
	median = np.nanpercentile(list_values, 50)
	mean = np.nanmean(list_values)
	std = np.nanstd(list_values)
	var = np.nanvar(list_values)
	rms = np.nanmean(np.sqrt(list_values ** 2))
	diff1 = np.diff(list_values, n=1)
	diff2 = np.diff(list_values, n=2)
	diff1_mean = np.nanmean(diff1)
	diff1_median = np.nanpercentile(diff1, 50)
	diff1_std = np.nanstd(diff1)
	diff2_mean = np.nanmean(diff2)
	diff2_median = np.nanpercentile(diff2, 50)
	diff2_std = np.nanstd(diff2)
	min_ratio = min(list_values) / len(list_values)
	max_ratio = max(list_values) / len(list_values)
	return [n5, n25, n75, n95, median, mean, std, var, rms, diff1_mean, diff1_median, diff1_std, diff2_mean,
			diff2_median, diff2_std, min_ratio, max_ratio]


# return [n5, n25, n75, n95, median, mean, std, var, rms]
# 过零率，即信号穿过0的次数
# 平均穿越率，即信号穿越平均值y的次数
def calculate_crossings(list_values):
	zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
	no_zero_crossings = len(zero_crossing_indices)
	mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
	no_mean_crossings = len(mean_crossing_indices)
	return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
	entropy = calculate_entropy(list_values)
	crossings = calculate_crossings(list_values)
	statistics = calculate_statistics(list_values)
	return [entropy] + crossings + statistics


# return [entropy] + statistics

def get_uci_har_features(dataset, labels, waveletname):
	uci_har_features = []
	for signal_no in range(0, len(dataset)):
		features = []
		for signal_comp in range(0, dataset.shape[2]):
			signal = dataset[signal_no, :, signal_comp]
			list_coeff = pywt.wavedec(signal, waveletname)
			for coeff in list_coeff:
				features += get_features(coeff)
		uci_har_features.append(features)
	X = np.array(uci_har_features)
	Y = np.array(labels)
	return X, Y


def get_train_test(df, y_col, x_cols, ratio):
	"""
	This method transforms a dataframe into a train and test set, for this you need to specify:
	1. the ratio train : test (usually 0.7)
	2. the column with the Y_values
	"""
	# mask = np.random.rand(len(df)) < ratio # 返回一组0-1之间均匀分布的随机值，小于ratio的为true，大于ratio的为false
	# df_train = df[mask]                    # 返回索引值为true的行
	# df_test = df[~mask]
	# df = df.sample(frac=1.0)
	# 打乱所有数据
	df = shuffle(df)
	df_train = df.iloc[:int(len(df) * ratio)]
	df_test = df.iloc[int(len(df) * ratio):]
	Y_train = df_train[y_col].values
	Y_test = df_test[y_col].values
	X_train = df_train[x_cols].values
	X_test = df_test[x_cols].values
	return df_train, df_test, X_train, Y_train, X_test, Y_test


filename = 'D:/pycharm/pythonProject/ECGclassify/ECGData/data1.mat'
ecg_data = sio.loadmat(filename)
ecg_signals = ecg_data['data1'][0][0][0]
ecg_labels_ = ecg_data['data1'][0][0][1]
ecg_labels = ecg_labels_
# ecg_labels = list(map(lambda x: x[0][0], ecg_labels_))# 取出标签ndarray的X[0][0],构成一个标签列表，成为真正的标签
# 读取出ecg信号的数据和标签 使用defaultdict构建一个默认value为list的字典


list_labels = ecg_labels
list_features = []

for signal in ecg_signals:
	features = []
	# list_coeff = pywt.wavedec(signal, 'db4')  #对信号进行小波变换，分成n个子带，这个n的大小由数据长度决定
	# for coeff in list_coeff:  # 对分成的每一个子带进行特征值的计算上述12个特征，在这个例子中，信号被分成了14个子带，所以一个信号最后得到12*14 = 168个特征，组成了这个信号的特征向量。
	features += get_features(signal)
	list_features.append(features)
df = pd.DataFrame(list_features)  # 将所有信号构成的特征向量列表变成dataframe表格形式

ycol = 'y'
xcols = list(range(df.shape[1]))  # 获取特征数量，这里是168个特征
# df.loc[:,ycol] = list_labels # 在dataframe的最后一列加上一个标签y，并将标签列表放到其中，这样特征向量和标签就对应起来并都在一个df中了
# df0 = df.loc[df['y'].isin(['0'])] #取出高兴，悲伤，愤怒，恐惧数据
# df1 = df.loc[df['y'].isin(['1'])]
# df2 = df.loc[df['y'].isin(['2'])]
# df3 = df.loc[df['y'].isin(['3'])]
# frames = [df3,df1]
#
# df_result= pd.concat(frames)
# df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_result, ycol, xcols, ratio = 0.7)
X_train,X_test, Y_train,  Y_test = train_test_split(df, list_labels, test_size=0.3, random_state=0,
													stratify=list_labels)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()
print(X_train.shape)
print(Y_train.shape)

# print((Y_train == 0).sum())
# print((Y_test == 0).sum())
# print((Y_train == 1).sum())
# print((Y_test == 1).sum())
# print((Y_train == 2).sum())
# print((Y_test == 2).sum())
# print((Y_train == 3).sum())
# print((Y_test == 3).sum())
cls = GradientBoostingClassifier(n_estimators=10,random_state=10)
cls.fit(X_train, Y_train) # 训练模型
train_score = cls.score(X_train, Y_train) #训练集得分
test_score = cls.score(X_test, Y_test) # 测试集得分
print("The Train Score is {:.5f}".format(train_score)) #数字格式化
print("The Test Score is {:.5f}".format(test_score))
print(Y_test)
Y_predict=cls.predict(X_test)
print(Y_predict)
# # #3.训练svm分类器
# #
# classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略
# classifier.fit(X_train,Y_train.ravel()) #ravel函数在降维时默认是行序优先
#
# print("训练集：",classifier.score(X_train, Y_train))
# print("测试集：",classifier.score(X_test, Y_test))
