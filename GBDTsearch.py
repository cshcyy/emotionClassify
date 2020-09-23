#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :GBDTsearch.py
# @Time        :2020/7/31  16:26
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
def gradient_boosting_classifier():
	# clf = GradientBoostingClassifier(n_estimators=280,max_depth=2,min_samples_split=86,max_features=8,learning_rate=0.05)
	clf = GradientBoostingClassifier(n_estimators=80,max_depth=1,min_samples_split=94,min_samples_leaf=45,max_features=8,learning_rate=0.05)

	return clf
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
	return [ n25, n75, median, mean, std, var, rms, diff1_mean, diff1_median, diff1_std, diff2_mean,
			diff2_median, diff2_std, min_ratio, max_ratio]



# return [n5, n25, n75, n95, median, mean, std, var, rms]
# 过零率，即信号穿过0的次数
# 平均穿越率，即信号穿越平均值y的次数
def calculate_crossings(list_values):
	zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
	no_zero_crossings = len(zero_crossing_indices)
	mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
	no_mean_crossings = len(mean_crossing_indices)
	return [no_mean_crossings]

def calculate_frequency(list_values):
	Fs = 5
	N = len(list_values)
	# print(N)
	# x = range(N)
	y = list_values
	# ff = np.arange(0, Fs, Fs / N)
	fft_y = fft(y)
	abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
	# angle_y = np.angle(fft_y)  # 取复数的角度
	normalization_y = abs_y / (N / 2)  # 归一化处理（双边频谱）
	normalization_y[0] = normalization_y[0] / 2  # 直流分量归一化应该除以N
	normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

	sum = np.sum(normalization_half_y)  # 总能量
	n5 = np.nanpercentile(normalization_half_y, 5)
	n25 = np.nanpercentile(normalization_half_y, 25)
	n75 = np.nanpercentile(normalization_half_y, 75)
	n95 = np.nanpercentile(normalization_half_y, 95)
	median = np.nanpercentile(normalization_half_y, 50)
	mean = np.nanmean(normalization_half_y)
	std = np.nanstd(normalization_half_y)
	var = np.nanvar(normalization_half_y)
	avg = np.mean(np.abs(normalization_half_y))
	rms = np.nanmean(np.sqrt(normalization_half_y ** 2))
	s = rms/avg #波形因子
	diff1 = np.diff(normalization_half_y, n=1)
	diff2 = np.diff(normalization_half_y, n=2)
	diff1_mean = np.nanmean(diff1)
	diff1_median = np.nanpercentile(diff1, 50)
	diff1_std = np.nanstd(diff1)
	diff2_mean = np.nanmean(diff2)
	diff2_median = np.nanpercentile(diff2, 50)
	diff2_std = np.nanstd(diff2)
	min_ratio = min(normalization_half_y) / len(normalization_half_y)
	max_ratio = max(normalization_half_y) / len(normalization_half_y)
	return [ sum, std, var, rms,diff1_mean, diff1_median, diff1_std, diff2_mean,
			diff2_median, diff2_std,min_ratio,max_ratio,s,mean,median]


def get_features(list_values):
	entropy = calculate_entropy(list_values)
	crossings = calculate_crossings(list_values)
	statistics = calculate_statistics(list_values)
	frequencyFeatures = calculate_frequency(list_values)
	# return [entropy] + crossings + statistics
	return crossings + statistics + frequencyFeatures
	return crossings + statistics

def getData_2():

	filename = 'D:/pycharm/pythonProject/ECGclassify/ECGData/data1.mat'
	ecg_data = sio.loadmat(filename)
	ecg_signals = ecg_data['data1'][0][0][0]
	ecg_labels_ = ecg_data['data1'][0][0][1]
	ecg_labels = ecg_labels_


	list_labels = ecg_labels
	list_features = []

	for signal in ecg_signals:
		features = []

		features += get_features(signal)
		list_features.append(features)
	df = pd.DataFrame(list_features)  # 将所有信号构成的特征向量列表变成dataframe表格形式

	ycol = 'y'
	xcols = list(range(df.shape[1]))  # 获取特征数量，这里是168个特征

	X_train, X_test, Y_train, Y_test = train_test_split(df, list_labels, test_size=0.3, random_state=0,
														stratify=list_labels)
	Y_train = Y_train.ravel()
	Y_test = Y_test.ravel()
	return (X_train, np.array(Y_train), X_test, np.array(Y_test))

def selectRFParam1():
	clf_GBDT =GradientBoostingClassifier(max_depth=2,min_samples_split=96 ,min_samples_leaf=45,max_features=8,learning_rate=0.05)
	param_grid = {"n_estimators": range(60, 90, 2)}
	# "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
	# "max_features": range(3,10),
	# "warm_start": [True, False],
	# "oob_score": [True, False],
	# "verbose": [True, False]}
	gsearch1 = GridSearchCV(clf_GBDT, param_grid=param_grid, n_jobs=4,cv=5)
	# start = time()
	T = getData_2()  # 获取数据集
	gsearch1.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	# 	  % (time() - start, len(grid_search.cv_results_['params'])))
	means = gsearch1.cv_results_['mean_test_score']
	params = gsearch1.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print("Test set score:{:.2f}".format(gsearch1.score(T[2],T[3])))
	print("Best parameters:{}".format(gsearch1.best_params_))
	print("Best score on train set:{:.2f}".format(gsearch1.best_score_))

# print(grid_search.cv_results_)
def selectRFParam2():
	clf_GBDT = GradientBoostingClassifier(n_estimators= 300)
	param_grid = {"max_depth": range(2, 3, 1),'min_samples_split':range(80,100,1)}
	gsearch1 = GridSearchCV(clf_GBDT, param_grid=param_grid, n_jobs=4,cv=5)
	T = getData_2()  # 获取数据集
	gsearch1.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	means = gsearch1.cv_results_['mean_test_score']
	params = gsearch1.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print("Test set score:{:.2f}".format(gsearch1.score(T[2],T[3])))
	print("Best parameters:{}".format(gsearch1.best_params_))
	print("Best score on train set:{:.2f}".format(gsearch1.best_score_))

def selectRFParam3():
	clf_GBDT = GradientBoostingClassifier(n_estimators=80,max_depth=2,min_samples_split=96 ,max_features=8,learning_rate=0.05 )
	param_grid = { 'min_samples_leaf':range(20,101,5)}
	gsearch1 = GridSearchCV(clf_GBDT, param_grid=param_grid, n_jobs=4,cv=5)
	T = getData_2()  # 获取数据集
	gsearch1.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	means = gsearch1.cv_results_['mean_test_score']
	params = gsearch1.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print("Test set score:{:.2f}".format(gsearch1.score(T[2],T[3])))
	print("Best parameters:{}".format(gsearch1.best_params_))
	print("Best score on train set:{:.2f}".format(gsearch1.best_score_))
def selectRFParam4():
	clf_GBDT = GradientBoostingClassifier(n_estimators=80,max_depth=2,min_samples_split=96 ,learning_rate=0.05 ,min_samples_leaf=45)
	param_grid = { 'max_features':range(1,19,1)}
	gsearch1 = GridSearchCV(clf_GBDT, param_grid=param_grid, n_jobs=4,cv=5)
	T = getData_2()  # 获取数据集
	gsearch1.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	means = gsearch1.cv_results_['mean_test_score']
	params = gsearch1.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print("Test set score:{:.2f}".format(gsearch1.score(T[2],T[3])))
	print("Best parameters:{}".format(gsearch1.best_params_))
	print("Best score on train set:{:.2f}".format(gsearch1.best_score_))

def selectRFParam5():
	clf_GBDT = GradientBoostingClassifier(n_estimators=80,max_depth=2,min_samples_split=96  ,min_samples_leaf=45,max_features=3)
	param_grid = { 'learning_rate':(0.1,0.05,0.01,0.005,0.0001)}
	gsearch1 = GridSearchCV(clf_GBDT, param_grid=param_grid, n_jobs=4,cv=5)
	T = getData_2()  # 获取数据集
	gsearch1.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	means = gsearch1.cv_results_['mean_test_score']
	params = gsearch1.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	print(gsearch1.best_params_, gsearch1.best_score_)
	print("Test set score:{:.2f}".format(gsearch1.score(T[2],T[3])))
	print("Best parameters:{}".format(gsearch1.best_params_))
	print("Best score on train set:{:.2f}".format(gsearch1.best_score_))
if __name__ == '__main__':
	(X_train, y_train,X_test, y_test) = getData_2()
	# 模型训练
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
														stratify=y_train)
	clf_GBDT = gradient_boosting_classifier()
	clf_GBDT.fit(X_train, y_train)

	y_pred = clf_GBDT.predict(X_test)
	print('识别率为: %.4f' % metrics.accuracy_score(y_test, y_pred))
	#
	# selectRFParam1()