# -*- coding: UTF-8 -*-
'''
	this module is to indentity handwriting by svm
	@author: Liu Weijie
'''
import svm
from numpy import *

# read data from .txt change to vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImage(dirName):
    import os
    fileList = os.listdir(dirName)
    featureList=[];labelList = []
    for file in fileList:
        if file[0] == '9': labelList.append(1)
        else: labelList.append(-1)
        featureList.append(img2vector(dirName+'/'+file).tolist())
    return featureList, labelList

def handwritingIndentify(kTup):
    featureList,labelList = loadImage('digits/trainingDigits')
    print featureList
    print labelList
    b, alphas = svm.smoP(array(featureList),labelList,200,0.0001,10000,kTup)
    print "b:",b, "alpha:",alphas
    svInd = nonzero(alphas.A > 0)[0]
    sVs = mat(array(featureList))[svInd]
    labelSV= mat(labelList).T[svInd]
    print "there a %s Support Vectors."%shape(sVs)[0]
    featureMat = mat(array(featureList));labelMat = mat(labelList).T
    m,n = shape(featureMat)
    errorCount = 0
    for i in range(m):
        kernelMat = svm.kernelTrans(sVs, featureMat[i,:], kTup)
        predict = kernelMat.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print "the training data error rate is %s"%(float(errorCount)/m)

if __name__ == '__main__':
    handwritingIndentify(('rbf',10))