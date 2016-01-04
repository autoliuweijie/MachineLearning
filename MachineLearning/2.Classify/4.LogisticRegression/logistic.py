# -*- coding: UTF-8 -*-
'''
	this module is logistic
	@author: Liu Weijie
'''
from numpy import *

# return mateix like featureMat and labelMat
def loadData(filename):
    featureList = [];labelList =[]
    fr = open(filename)
    for dataLine in fr.readlines():
        dataList = dataLine.strip().split()
        tempList = []
        for data in dataList:
            tempList.append(float(data))
        tempList.pop(-1)
        featureList.append(tempList)
        labelList.append(float(dataList[-1]))
    return mat(featureList), mat(labelList)

# sigmoid function
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# caclulate logistic parameter w by grad method
# return list like weights
def gradAscent(featureMat, labelMat, numIter = 150):
    dataMatrix = featureMat.tolist()
    classLabels = labelMat.T.tolist()
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# test logistiv regress
def testLogistic():
    trainFeatureMat, trainLabelMat = loadData('horseColicTraining.txt')
    weights = gradAscent(trainFeatureMat, trainLabelMat)
    testFeatureMat, testLabelMat = loadData('horseColicTest.txt')
    errorCount = 0
    m,n = shape(testFeatureMat)
    for i in range(m):
        inX = (testFeatureMat[i]*mat(weights).T)[0,0]
        predict = sigmoid(inX)
        if predict > 0.5 : predict = 1.0
        if predict == testLabelMat[0,i]:
            errorCount += 1
    print "error rate = %s"%(float(errorCount)/m)

def main():
    featureMat, labelMat = loadData('horseColicTraining.txt')
    print featureMat, labelMat

if __name__ == '__main__':
    testLogistic()