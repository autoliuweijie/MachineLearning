# -*- coding: utf-8 -*-
#this is algorithm of kNN
from numpy import *
import operator
import matplotlib.pyplot as plt

#data for test
def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

#function : kNN
#imput    : inX － list - vector need to be classified
#           dataSet - array - character array 
#           labels - list - labels list
#           k - int - k in kNN
#return   : string - classify result 
def classify_kNN(inX, dataSet, labels, k):
	# calculate the distance to every sample
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistance = sqDiffMat.sum(axis = 1)
	distance = sqDistance**0.5
	# sort the distance
	sortedDistIndicies = distance.argsort()

	classCount= {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

#import data from .txt into numpy data type
def file2matrix(filename):
	fr = open(filename)
	arrayLines = fr.readlines()
	numberOfLines = len(arrayLines)
	returnLabel = [] 
	returnMat = zeros((numberOfLines,3))
	for index, line in enumerate(arrayLines):
		line = line.strip().split()
		returnMat[index,:] = line[0:3]
		returnLabel.append(line[-1])
	return returnMat, returnLabel

#convert data to norm type in book
def autoNorm(dataSet):
	maxData = dataSet.max(0)
	minData = dataSet.min(0)
	dataLenth = dataSet.shape[0]
	normDataSet = (dataSet - tile(minData,(dataLenth,1))) /(tile((maxData - minData), (dataLenth,1) ))
	return normDataSet, maxData, minData

#convert labels to number
def labels2number(labels):
	labelsDict = {}
	labelsNumber = 1
	for nowLabel in labels:
		if not (nowLabel in labelsDict):
			labelsDict[nowLabel] = labelsNumber
			labelsNumber += 1

	returnList = []
	for nowLabel in labels:
		returnList.append(labelsDict[nowLabel])
	return returnList

#test classify methd
def dataingClassTest(funname):
	group, labels = file2matrix('datingTestSet.txt')
	group, maxData, minData = autoNorm(group)
	labels = labels2number(labels)

	dataLenth = group.shape[0]
	trainLenth = int(0.9 * dataLenth)
	testLenth = dataLenth - trainLenth
	trainGroup = group[:trainLenth,:]
	testGroup = group[trainLenth:, :]
	trainLabel = labels[:trainLenth]
	testLabel = labels[trainLenth:]

	trueNumber = 0
	for i in range(testLenth):
		trueLabel = testLabel[i]
		classifyLabel = funname(testGroup[i,:],trainGroup,trainLabel,3)
		if trueLabel == classifyLabel:
			trueNumber += 1
		print "Real Label: %s  Classify Label %s"%(trueLabel,classifyLabel)
	print "true rate: ", float(trueNumber)/testLenth

#classify person
def classifyPerson(funname):
	data1 = float(raw_input("Please input data1"))
	data2 = float(raw_input("Please input data2"))
	data3 = float(raw_input("Please input data3"))
	group, labels = file2matrix('datingTestSet.txt')
	group, maxData, minData = autoNorm(group)

	personData = (array([data1, data2, data3]) - minData )/(maxData - minData)
	personLabels = funname(personData.tolist(), group, labels, 3)
	print "This person is", personLabels


#test
if __name__ == "__main__":
	classifyPerson(classify_kNN)
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# ax.scatter(group[:,0],group[:,1], 15*array(labels), 15*array(labels))
	# plt.show()
