#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :ECGclassify
# @FileName    :multiclassify.py
# @Time        :2020/7/24  16:41
# @Author      :Shuhao Chen
import os
import numpy as np
import pandas as pd
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


# 读取sklearn自带的数据集（鸢尾花）
def getData_1():
	iris = datasets.load_iris()
	X = iris.data  # 样本特征矩阵，150*4矩阵，每行一个样本，每个样本维度是4
	y = iris.target  # 样本类别矩阵，150维行向量，每个元素代表一个样本的类别


# 读取本地excel表格内的数据集（抽取每类60%样本组成训练集，剩余样本组成测试集）
# 返回一个元祖，其内有4个元素（类型均为numpy.ndarray）：
# （1）归一化后的训练集矩阵，每行为一个训练样本，矩阵行数=训练样本总数，矩阵列数=每个训练样本的特征数
# （2）每个训练样本的类标
# （3）归一化后的测试集矩阵，每行为一个测试样本，矩阵行数=测试样本总数，矩阵列数=每个测试样本的特征数
# （4）每个测试样本的类标
# 【注】归一化采用“最大最小值”方法。
def getData_2():
	# fPath = 'D:\分类算法\binary_classify_data.txt'
	# if os.path.exists(fPath):
	# 	data = pd.read_csv(fPath, header=None, skiprows=1, names=['class0', 'pixel0', 'pixel1', 'pixel2', 'pixel3'])
	# 	X_train1, X_test1, y_train1, y_test1 = train_test_split(data, data['class0'], test_size=0.4, random_state=0)
	# 	min_max_scaler = preprocessing.MinMaxScaler()  # 归一化
	# 	X_train_minmax = min_max_scaler.fit_transform(np.array(X_train1))
	# 	X_test_minmax = min_max_scaler.fit_transform(np.array(X_test1))
	# 	return (X_train_minmax, np.array(y_train1), X_test_minmax, np.array(y_test1))
	# else:
	# 	print('No such file or directory!')
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
	X_train, X_test, Y_train, Y_test = train_test_split(df, list_labels, test_size=0.3, random_state=0,
														stratify=list_labels)
	Y_train = Y_train.ravel()
	Y_test = Y_test.ravel()
	return (X_train, np.array(Y_train), X_test, np.array(Y_test))


# 读取本地excel表格内的数据集（每类随机生成K个训练集和测试集的组合）
# 【K的含义】假设一共有1000个样本，K取10，那么就将这1000个样本切分10份（一份100个），那么就产生了10个测试集
# 对于每一份的测试集，剩余900个样本即作为训练集
# 结果返回一个字典：键为集合编号（1train, 1trainclass, 1test, 1testclass, 2train, 2trainclass, 2test, 2testclass...），值为数据
# 其中1train和1test为随机生成的第一组训练集和测试集（1trainclass和1testclass为训练样本类别和测试样本类别），其他以此类推
def getData_3():
	fPath = 'D:\\分类算法\\binary_classify_data.txt'
	if os.path.exists(fPath):
		# 读取csv文件内的数据，
		dataMatrix = np.array(
			pd.read_csv(fPath, header=None, skiprows=1, names=['class0', 'pixel0', 'pixel1', 'pixel2', 'pixel3']))
		# 获取每个样本的特征以及类标
		rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
		sampleData = []
		sampleClass = []
		for i in range(0, rowNum):
			tempList = list(dataMatrix[i, :])
			sampleClass.append(tempList[0])
			sampleData.append(tempList[1:])
		sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
		classM = np.array(sampleClass)  # 一维列向量，每个元素对应每个样本所属类别
		# 调用StratifiedKFold方法生成训练集和测试集
		skf = StratifiedKFold(n_splits=10)
		setDict = {}  # 创建字典，用于存储生成的训练集和测试集
		count = 1
		for trainI, testI in skf.split(sampleM, classM):
			trainSTemp = []  # 用于存储当前循环抽取出的训练样本数据
			trainCTemp = []  # 用于存储当前循环抽取出的训练样本类标
			testSTemp = []  # 用于存储当前循环抽取出的测试样本数据
			testCTemp = []  # 用于存储当前循环抽取出的测试样本类标
			# 生成训练集
			trainIndex = list(trainI)
			for t1 in range(0, len(trainIndex)):
				trainNum = trainIndex[t1]
				trainSTemp.append(list(sampleM[trainNum, :]))
				trainCTemp.append(list(classM)[trainNum])
			setDict[str(count) + 'train'] = np.array(trainSTemp)
			setDict[str(count) + 'trainclass'] = np.array(trainCTemp)
			# 生成测试集
			testIndex = list(testI)
			for t2 in range(0, len(testIndex)):
				testNum = testIndex[t2]
				testSTemp.append(list(sampleM[testNum, :]))
				testCTemp.append(list(classM)[testNum])
			setDict[str(count) + 'test'] = np.array(testSTemp)
			setDict[str(count) + 'testclass'] = np.array(testCTemp)
			count += 1
		return setDict
	else:
		print('No such file or directory!')


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


# report函数，将调参的详细结果存储到本地F盘（路径可自行修改，其中n_top是指定输出前多少个最优参数组合以及该组合的模型得分）
def report(results, n_top=5488):
	f = open('F:/grid_search_rf.txt', 'w')
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			f.write("Model with rank: {0}".format(i) + '\n')
			f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				results['mean_test_score'][candidate],
				results['std_test_score'][candidate]) + '\n')
			f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
			f.write("\n")
	f.close()


# 自动调参（以随机森林为例）
def selectRFParam():
	clf_RF = RF()
	param_grid = {"max_depth": [3, 15],
				  "min_samples_split": [3, 5, 10],
				  "min_samples_leaf": [3, 5, 10],
				  "bootstrap": [True, False],
				  "criterion": ["gini", "entropy"],
				  "n_estimators": range(10, 50, 10)}
	# "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
	# "max_features": range(3,10),
	# "warm_start": [True, False],
	# "oob_score": [True, False],
	# "verbose": [True, False]}
	grid_search = GridSearchCV(clf_RF, param_grid=param_grid, n_jobs=4)
	# start = time()
	T = getData_2()  # 获取数据集
	grid_search.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
	# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
	# 	  % (time() - start, len(grid_search.cv_results_['params'])))
	report(grid_search.cv_results_)


# print(grid_search.cv_results_)


# “主”函数1（KFold方法生成K个训练集和测试集，即数据集采用getData_3()函数获取，计算这K个组合的平均识别率）
def totalAlgorithm_1():
	# 获取各个分类器
	clf_KNN = KNN()
	clf_LDA = LDA()
	clf_SVM = SVM()
	clf_LR = LR()
	clf_RF = RF()
	clf_NBC = native_bayes_classifier()
	clf_DTC = decision_tree_classifier()
	clf_GBDT = gradient_boosting_classifier()
	# 获取训练集和测试集
	setDict = getData_3()
	setNums = len(setDict.keys()) / 4  # 一共生成了setNums个训练集和setNums个测试集，它们之间是一一对应关系
	# 定义变量，用于将每个分类器的所有识别率累加
	KNN_rate = 0.0
	LDA_rate = 0.0
	SVM_rate = 0.0
	LR_rate = 0.0
	RF_rate = 0.0
	NBC_rate = 0.0
	DTC_rate = 0.0
	GBDT_rate = 0.0
	for i in range(1, int(setNums + 1)):
		trainMatrix = setDict[str(i) + 'train']
		trainClass = setDict[str(i) + 'trainclass']
		testMatrix = setDict[str(i) + 'test']
		testClass = setDict[str(i) + 'testclass']
		# 输入训练样本
		clf_KNN.fit(trainMatrix, trainClass)
		clf_LDA.fit(trainMatrix, trainClass)
		clf_SVM.fit(trainMatrix, trainClass)
		clf_LR.fit(trainMatrix, trainClass)
		clf_RF.fit(trainMatrix, trainClass)
		clf_NBC.fit(trainMatrix, trainClass)
		clf_DTC.fit(trainMatrix, trainClass)
		clf_GBDT.fit(trainMatrix, trainClass)
		# 计算识别率
		KNN_rate += getRecognitionRate(clf_KNN.predict(testMatrix), testClass)
		LDA_rate += getRecognitionRate(clf_LDA.predict(testMatrix), testClass)
		SVM_rate += getRecognitionRate(clf_SVM.predict(testMatrix), testClass)
		LR_rate += getRecognitionRate(clf_LR.predict(testMatrix), testClass)
		RF_rate += getRecognitionRate(clf_RF.predict(testMatrix), testClass)
		NBC_rate += getRecognitionRate(clf_NBC.predict(testMatrix), testClass)
		DTC_rate += getRecognitionRate(clf_DTC.predict(testMatrix), testClass)
		GBDT_rate += getRecognitionRate(clf_GBDT.predict(testMatrix), testClass)
	# 输出各个分类器的平均识别率（K个训练集测试集，计算平均）
	print
	print
	print
	print('K Nearest Neighbor mean recognition rate: ', KNN_rate / float(setNums))
	print('Linear Discriminant Analysis mean recognition rate: ', LDA_rate / float(setNums))
	print('Support Vector Machine mean recognition rate: ', SVM_rate / float(setNums))
	print('Logistic Regression mean recognition rate: ', LR_rate / float(setNums))
	print('Random Forest mean recognition rate: ', RF_rate / float(setNums))
	print('Native Bayes Classifier mean recognition rate: ', NBC_rate / float(setNums))
	print('Decision Tree Classifier mean recognition rate: ', DTC_rate / float(setNums))
	print('Gradient Boosting Decision Tree mean recognition rate: ', GBDT_rate / float(setNums))


# “主”函数2（每类前x%作为训练集，剩余作为测试集，即数据集用getData_2()方法获取，计算识别率）
def totalAlgorithm_2():
	# 获取各个分类器
	featureName = ['no_mean_crossings', 'n5', 'n25', 'n75', 'n95', 'median', 'mean', 'std', 'var', 'rms', 'diff1_mean',
				   'diff1_median', 'diff1_std', 'diff2_mean', 'diff2_median', 'diff2_std', 'min_ratio', 'max_ratio']
	clf_KNN = KNN()
	clf_LDA = LDA()
	clf_SVM = SVM()
	clf_LR = LR()
	clf_RF = RF()
	clf_NBC = native_bayes_classifier()
	clf_DTC = decision_tree_classifier()
	clf_GBDT = gradient_boosting_classifier()
	# 获取训练集和测试集
	T = getData_2()
	trainMatrix, trainClass, testMatrix, testClass = T[0], T[1], T[2], T[3]
	# 输入训练样本
	clf_KNN.fit(trainMatrix, trainClass)
	clf_LDA.fit(trainMatrix, trainClass)
	clf_SVM.fit(trainMatrix, trainClass)
	clf_LR.fit(trainMatrix, trainClass)
	clf_RF.fit(trainMatrix, trainClass)
	# clf_NBC.fit(trainMatrix, trainClass)
	clf_DTC.fit(trainMatrix, trainClass)
	clf_GBDT.fit(trainMatrix, trainClass)
	# 输出各个分类器的识别率
	importances = clf_RF.feature_importances_
	print("重要性：", importances)
	feat_labels = featureName
	x_columns = featureName
	indices = np.argsort(importances)[::-1]
	for f in range(trainMatrix.shape[1]):
		# 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
		# 到根，根部重要程度高于叶子。
		print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))  # %-*s -号代表左对齐、后补空白，*号代表对齐宽度由输入时确定，s代表输入字符串
	print("重要性：", importances)
	print('K Nearest Neighbor recognition rate: ', getRecognitionRate(clf_KNN.predict(testMatrix), testClass))
	print('Linear Discriminant Analysis recognition rate: ', getRecognitionRate(clf_LDA.predict(testMatrix), testClass))
	print('Support Vector Machine recognition rate: ', getRecognitionRate(clf_SVM.predict(testMatrix), testClass))
	print('Logistic Regression recognition rate: ', getRecognitionRate(clf_LR.predict(testMatrix), testClass))
	print('Random Forest recognition rate: ', getRecognitionRate(clf_RF.predict(testMatrix), testClass))
	# print('Native Bayes Classifier recognition rate: ', getRecognitionRate(clf_NBC.predict(testMatrix), testClass))
	print('Decision Tree Classifier recognition rate: ', getRecognitionRate(clf_DTC.predict(testMatrix), testClass))
	print('Gradient Boosting Decision Tree recognition rate: ',
		  getRecognitionRate(clf_GBDT.predict(testMatrix), testClass))


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


if __name__ == '__main__':
	# print('K个训练集和测试集的平均识别率')
	# totalAlgorithm_1()
	print('每类前x%训练，剩余测试，各个模型的识别率')
	totalAlgorithm_2()
# selectRFParam()
# print('随机森林参数调优完成！')
