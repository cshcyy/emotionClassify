#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :gridSearch.py
# @Time        :2020/7/27  14:39
# @Author      :Shuhao Chen
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
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

# K近邻（K Nearest Neighbor）
def KNN():
	clf = neighbors.KNeighborsClassifier()
	return clf


# 线性鉴别分析（Linear Discriminant Analysis）
def LDA():
	clf = LinearDiscriminantAnalysis()
	return clf


# 支持向量机（Support Vector Machine）
def SVM():
	clf = svm.SVC()
	return clf


# 逻辑回归（Logistic Regression）
def LR():
	clf = LogisticRegression()
	return clf


# 随机森林决策树（Random Forest）
def RF():
	clf = RandomForestClassifier()
	return clf


# 多项式朴素贝叶斯分类器
def native_bayes_classifier():
	clf = MultinomialNB(alpha=0.01)
	return clf


# 决策树
def decision_tree_classifier():
	clf = tree.DecisionTreeClassifier()
	return clf


# GBDT
def gradient_boosting_classifier():
	clf = GradientBoostingClassifier(n_estimators=200)
	return clf


# 计算识别率
def getRecognitionRate(testPre, testClass):
	testNum = len(testPre)
	rightNum = 0
	for i in range(0, testNum):
		if testClass[i] == testPre[i]:
			rightNum += 1
	return float(rightNum) / float(testNum)
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
	return [no_mean_crossings]


def get_features(list_values):
	entropy = calculate_entropy(list_values)
	crossings = calculate_crossings(list_values)
	statistics = calculate_statistics(list_values)
	# return [entropy] + crossings + statistics
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

def gridsearch_cv(model, test_param, X, y, cv=5):
    gsearch = GridSearchCV(estimator=model, param_grid=test_param, scoring='roc_auc', n_jobs=4, iid=False, cv=cv)
    gsearch.fit(X, y)
    print('CV Results: ', gsearch.cv_results_)
    print('Best Params: ', gsearch.best_params_)
    print('Best Score: ', gsearch.best_score_)
    return gsearch.best_params_


if __name__ == '__main__':
	(X_train, y_train,X_test, y_test) = getData_2()
	# 模型训练
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
														stratify=y_train)
	model = XGBClassifier(n_estimators=200)
	# model = GradientBoostingClassifier(n_estimators=200)
	eval_set = [(X_val, y_val)]
	model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True ,eval_metric="merror")
	y_pred = model.predict(X_test)
	y_score = model.predict_proba(X_test)  #输出预测概率
	## 获取onehot编码
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.preprocessing import LabelEncoder
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_test)
	# print(integer_encoded)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	# print(onehot_encoded)


	# print(y_score)
	print(y_pred)
	print(y_test)
	precision_every_class = metrics.precision_score(y_test, y_pred, average=None)
	precision_macro = metrics.precision_score(y_test, y_pred, average='macro')
	precision_micro = metrics.precision_score(y_test, y_pred, average='micro')
	precision_weighted = metrics.precision_score(y_test, y_pred, average='weighted')
	f1_every_class = metrics.f1_score(y_test, y_pred, average=None)
	f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
	f1_micro = metrics.f1_score(y_test, y_pred, average='micro')

	recall_every_class = metrics.recall_score(y_test, y_pred, average=None)
	recall_macro = metrics.recall_score(y_test, y_pred, average='macro')
	recall_micro = metrics.recall_score(y_test, y_pred, average='micro')

	# print('AUC: %.4f' % metrics.roc_auc_score(onehot_encoded, y_score, average=None))
	print('AUC: %.4f' % metrics.roc_auc_score(onehot_encoded, y_score,average='micro'))
	print('AUC: %.4f' % metrics.roc_auc_score(onehot_encoded, y_score, average='macro'))
	print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
	print('precision: ', precision_every_class,precision_macro,precision_micro,precision_weighted)
	print('F1: ',f1_every_class,f1_macro,f1_micro)
	print('recall: ',recall_every_class,recall_macro,recall_micro)
	# print('Recall: %.4f' % metrics.recall_score(y_test, y_pred,average='micro'))
	# # print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
	# # print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))
	# metrics.confusion_matrix(y_test, y_pred)
	# print("Accuracy: %.2f%%" % (accuracy * 100.0))
	# plot_importance(model)
	# pyplot.show()
	# parameters = {
	# 	'max_depth': [5, 10, 15, 20, 25],
	# 	'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
	# 	'n_estimators': [50, 100, 200, 300, 500],
	# 	'min_child_weight': [0, 2, 5, 10, 20],
	# 	'max_delta_step': [0, 0.2, 0.6, 1, 2],
	# 	'subsample': [0.6, 0.7, 0.8, 0.85, 0.95]
	#
	#
	# }
	# grid_search = GridSearchCV(model, param_grid=parameters, scoring='accuracy', cv=3)
	# grid_search.fit(X_train, y_train)
	# print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
	# print("Best parameters:{}".format(grid_search.best_params_))
	# print("Best score on train set:{:.2f}".format(grid_search.best_score_))

