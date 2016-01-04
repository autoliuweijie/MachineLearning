# -*- coding: UTF-8 -*-
'''
	this module is decision tree
	@author: Liu Weijie
'''
import  math
import operator
from drawATree import *
import pickle

# create dataSet
def createDataSet():
	dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	feature = ['no surfacing', 'flippers']
	return dataSet, feature

# calculate ShannonEnt
def calcShannonEnt(dataSet):
	dataLen = len(dataSet)
	classCount = {}
	for dataLine in dataSet:
		classCount[dataLine[-1]] = classCount.get(dataLine[-1],0) + 1
	Ent = 0
	for key in classCount:
		p = float(classCount[key])/dataLen
		logP = math.log(p,2)
		Ent = Ent - p*logP
	return Ent

# split dataSet
def splitDataSet(dataSet, axis, value):
	returnDataSet = []
	for line in dataSet:
		if line[axis] == value:
			temLine = line[:axis]
			temLine.extend(line[axis+1:])
			returnDataSet.append(temLine)
	return returnDataSet

# choose best feature to split
def chooseBestFeatureToSplit(dataSet):
	featureLen = len(dataSet[0])-1
	dataLen = len(dataSet)
	bestFeature = 0
	maxEntGrep = 0
	baseEnt = calcShannonEnt(dataSet)
	for axis in range(featureLen):
		featureValueList = set([ feature[axis] for feature in dataSet])
		ent = 0.0
		for i in featureValueList:
			tempDataSet = splitDataSet(dataSet, axis, i)
			tempEnt = calcShannonEnt(tempDataSet)
			ent = ent + (float(len(tempDataSet))/dataLen)*tempEnt
		entGrep = baseEnt - ent
		if entGrep > maxEntGrep:
			maxEntGrep = entGrep
			bestFeature = axis
	return bestFeature

# select majority class
def majorityCnt(dataSet):
	classCount = {}
	for dataLine in dataSet:
		classCount[dataLine[-1]] = classCount.get(dataLine[-1],0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

# create trees
def createTree(dataSet, feature):
	thisFeature = feature[:]
	classList = [tempLabel[-1] for tempLabel in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(dataSet)

	splitFeatureIndex = chooseBestFeatureToSplit(dataSet)
	splitFeature = thisFeature[splitFeatureIndex]
	returnTree = {splitFeature: {} }
	del(thisFeature[splitFeatureIndex])

	splitFeatureList = [tempFeature[splitFeatureIndex ] for tempFeature in dataSet]
	splitFeatureValueList = list(set(splitFeatureList))
	for featureValue in splitFeatureValueList:
		newDataSet = splitDataSet(dataSet, splitFeatureIndex , featureValue)
		returnTree[splitFeature ][featureValue] = createTree(newDataSet,thisFeature)
	return returnTree

# save trees
def saveMyTree(tree, filename):
	fw = open(filename, 'w')
	pickle.dump(tree, fw)
	fw.close()

# load my tree
def loadMyTree(filename):
	fr = open(filename)
	return pickle.load(fr)

# classify by tree
def classifyByTree(tree, featureLabel, data):
	firstStr = tree.keys()[0]
	featureIndex = featureLabel.index(firstStr)
	secondList = tree[firstStr]
	maybeTree = secondList[data[featureIndex]]
	if type(maybeTree).__name__ == 'dict':
		returnLabel = classifyByTree(maybeTree, featureLabel, data)
	else:
		returnLabel = maybeTree
	return returnLabel

# test
if __name__ == '__main__':
	dataSet, feature = createDataSet()
	myTree = createTree(dataSet,feature)
	# saveMyTree(myTree,'myTree')
	# myTree = loadMyTree('myTree')
	drawTree(myTree)
	result = classifyByTree(myTree, feature, [0,1])
	print "This is ",result



















