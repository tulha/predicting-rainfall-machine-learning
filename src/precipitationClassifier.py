import numpy as np 
from operator import itemgetter
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA

def parseFiles():
	idx = 0
	filename = ""
	training = np.zeros([250, 12])
	for i in range(1,251):
		filename = "Data/new/" + str(i) + ".txt"
		with open(filename) as f:
			trainPoint = parseLine(f)
			training[idx] = trainPoint
			idx += 1
	np.savetxt('Data/new/training.txt', training)

def parseLine(f):
	idx = 0
	trainPoint = []
	for line in f:
		if idx == 1:
			trainPoint.append(float(line.split(" ")[-3].split("=")[1]))
		elif idx == 10 or idx == 11 or idx == 12:
			#skip
			idx = idx		
		else:
			trainPoint.append(float(line.split(" ")[-2].split("=")[1]))
		idx += 1

	return np.asarray(trainPoint)

def kNearestNeighbors(trainingData, trainingLabels, testData, testLabels, k):
	correct = 0
	predictedLabels = []
	dist = np.zeros(trainingData.shape[0]) 
	for test in range(0,testData.shape[0]):
		for train in range(0,trainingData.shape[0]):
			sample = testData[test]
			dist[train] = np.linalg.norm(sample - trainingData[train])
		minIndices = np.argpartition(dist, k)[:k]
		ones = 0
		zeros = 0
		for K in minIndices:
			if trainingLabels[K] == 1:
				ones += 1
			else:
				zeros += 1
		if(ones > zeros):
			classify = 1
		else:
			classify = 0
		dist = np.zeros(trainingData.shape[0])
		predictedLabels.append(classify)
		if classify == testLabels[test]:
			correct = correct + 1
		else:
			dummy = 1
	acc = (correct/testData.shape[0])*100
	TP_rate, FP_rate = assembleROC(predictedLabels, testLabels)

	return acc, TP_rate, FP_rate 

def returnKfolds(splits, trainingData, labels):
	trainingSets = []
	testingSets = []
	kf = KFold(n_splits=splits, shuffle=True)
	kf.get_n_splits(trainingData)
	idx = 1
	for train_index, test_index in kf.split(trainingData):
		X_train, X_test = trainingData[train_index], trainingData[test_index]
		y_train, y_test = labels[train_index], labels[test_index]
		trainingSets.append([X_train, y_train])
		testingSets.append([X_test, y_test])
		idx += 1
	if(X_train.shape[0]+X_test.shape[0]==trainingData.shape[0]==y_train.shape[0]+y_test.shape[0]):
		print(splits,"splits of the input data were successful.")
	return trainingSets, testingSets

def normalize(columnList):
	minimum = min(columnList)
	maximum = max(columnList)
	newColumnList = [2*((x-minimum)/maximum-minimum)-1 for x in columnList]
	# print(newColumnList)
	return newColumnList


def assembleROC(predictedLabels, trueLabels):
	TP, TN, FP, FN = 0, 0, 0, 0
	for idx in range(0, len(predictedLabels)):
		if predictedLabels[idx] == 1 and trueLabels[idx] == 1:
			TP += 1
		elif predictedLabels[idx] == 0 and trueLabels[idx] == 0:
			TN += 1
		elif predictedLabels[idx] == 1 and trueLabels[idx] == 0:
			FP += 1
		elif predictedLabels[idx] == 0 and trueLabels[idx] == 1:
			FN += 1
	TP_rate = TP/(TP + FN)
	FP_rate = FP/(FP + TN)

	return TP_rate, FP_rate

def plotkNN():
	TP = [0.8, 0.7142857142857143, 0.25, 0.6666666666666666, 0.5555555555555556, 0.4444444444444444, 0.5833333333333334, 0.6363636363636364, 0.7, 0.8] 
	FP = [0.2, 0.0, 0.09523809523809523, 0.125, 0.0625, 0.0, 0.0, 0.07142857142857142, 0.06666666666666667, 0.3]
	k_ = [65.6, 78.8, 77.6, 78.0, 79.2, 80.0, 80.0, 81.2, 76.0, 78.8, 76.8, 77.6, 76.0, 77.6, 76.8, 78.4, 77.6, 78.0, 77.6, 79.2, 78.4, 78.4, 79.2, 78.4, 77.2, 77.6, 76.8, 77.2, 78.0, 78.0, 76.8, 76.4, 76.8, 78.4, 77.6, 77.6, 77.2, 77.6, 77.2, 77.2, 76.0, 77.2, 76.4, 77.6, 77.2, 78.4, 76.8, 76.8, 77.6, 78.4, 76.8, 78.8, 78.4, 79.2, 78.4, 78.8, 78.4, 79.2, 79.6, 79.2, 78.4, 79.2, 77.6, 78.4, 78.0, 78.8, 77.6, 77.6, 77.2, 78.8, 77.2, 77.2, 78.0, 78.4, 78.0, 77.2, 76.4, 78.0, 76.0, 76.4, 75.6, 76.8, 74.8, 76.0, 76.0, 75.6, 74.0, 76.4, 73.6, 74.4, 73.2, 74.0, 71.6, 72.0, 70.4, 71.2, 72.0, 72.0, 70.8, 72.0, 70.8, 70.8, 71.2, 70.4, 70.4, 70.8, 70.4, 70.8, 68.8, 70.8, 68.8, 70.0, 68.4, 68.8, 68.0, 69.2, 68.0, 68.8, 67.6, 68.4, 68.0, 68.0, 67.6, 68.0, 67.2, 67.6, 66.4, 67.2, 66.8, 67.2, 66.4, 66.4, 65.6, 66.4, 65.6, 65.6, 66.0, 66.0, 66.0, 66.0, 65.6, 65.6, 65.6, 65.6, 66.0, 66.0, 66.0, 66.0, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6, 65.6]
	k = list(range(201))
	print("The optimal value of k is: ", max(k_))
	z = np.polyfit(k, k_, 14)
	f = np.poly1d(z)
	x_new = np.linspace(k[0], k[-1], 50)
	y_new = f(k)
	
	# plt.plot(FP, TP, label='kNN ROC Curve', markersize=4)
	plt.plot(k, y_new, 'b')
	plt.plot(k, k_, 'o', label='Number of neigbors k', markersize=4)
	plt.ylabel('% Accuracy (10 fold)')
	plt.xlabel('Number of neighbors, k')
	plt.legend()
	plt.show()

def plotknnROC(TP_rate, FP_rate):
	plt.plot(FP_rate, TP_rate, 'o', label='Best Fit ROC Curve', markersize=4)
	plt.ylabel('TP Rate')
	plt.xlabel('FP Rate')
	plt.title("kNN ROC Curve (Orlando)")
	plt.legend()
	plt.show()

def knn(trainingSets, testingSets):
	for numSet in range(0,10):
		term = kNearestNeighbors(trainingSets[numSet][0], trainingSets[numSet][1], testingSets[numSet][0], testingSets[numSet][1], K)
		temp.append(term[0])
		TPArr.append(term[1])
		FPArr.append(term[2])

def SVMClassifier(trainingData, trainingLabels, testData, testLabels):
	clf = SVC()
	clf.fit(trainingData, trainingLabels)
	predictedLabels = clf.predict(testData)
	TPRATE, FPRATE = assembleROC(predictedLabels, testLabels)

	correct = 0
	for i in range(0, len(testLabels)):
		if(predictedLabels[i] == testLabels[i]):
			correct += 1


	return (correct/len(testLabels))*100, TPRATE, FPRATE

def applyPCA(X, numFeatures):
	# Here we will apply our dimensionalty reuction algorithm. 
	pca = PCA(n_components=numFeatures, svd_solver='full', whiten=True).fit(X)
	return pca.transform(X)

def dimensionalAnalysis(trainingSets, testingSets, numFeatures):
	K = 7
	temp1 = []
	temp2 = []
	svmAcc = []
	knnAcc = []

	for numFeatures in range(1,13):
		for numSet in range(0,10):
			term1 = SVMClassifier(applyPCA(trainingSets[numSet][0], numFeatures), trainingSets[numSet][1], applyPCA(testingSets[numSet][0], numFeatures), testingSets[numSet][1])
			term2 = kNearestNeighbors(applyPCA(trainingSets[numSet][0], numFeatures), trainingSets[numSet][1], applyPCA(testingSets[numSet][0], numFeatures), testingSets[numSet][1], K)
			temp1.append(term1[0])
			temp2.append(term2[0])
		svmAcc.append(np.mean(temp1))
		knnAcc.append(np.mean(temp2))
		temp1 = []
		temp2 = []

	return knnAcc, svmAcc

# parseFiles()
trainingData = np.loadtxt("Data/new/training.txt")
labels = np.loadtxt("Data/new/labels.txt")
a,b = np.shape(trainingData)
normalizedData = np.zeros(shape=(a,b))
min_max_scaler = preprocessing.MinMaxScaler()
normalizedData = min_max_scaler.fit_transform(trainingData)
trainingSets, testingSets = returnKfolds(10, normalizedData, labels)

svmAcc = [69.6, 66.0, 67.2, 63.6, 62.8, 64.8, 63.2, 61.2, 63.2, 61.6, 63.6, 57.6]
knnAcc = [68.0, 67.2, 66.0, 64.8, 64.0, 65.2, 62.8, 62.4, 62.0, 61.2, 62.4, 61.6]
print(svmAcc)
print(knnAcc)
v = [1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(v, svmAcc,'o')
plt.plot(v, knnAcc,'x')
plt.title("KNN and SVM Dimensional Analysis")
plt.show()
pca = PCA(n_components=12, svd_solver='full', whiten=True).fit(normalizedData)
print(pca.explained_variance_ratio_)