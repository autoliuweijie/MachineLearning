# -*- coding: UTF-8 -*-
'''
	this module is bayes
	@author: Liu Weijie
'''

import numpy

#create text vector
def createTextVector():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#get charactor label vector
#input  trainTextVector: List
#output charactorLabelVector : charactor label vector fof goalTextVector, List
#       charactorValueVector : charactor value vector fof goalTextVector, List
def getCharactorLabelVector(trainTextVector):
     textAssemble = set([])
     for trainTextVectorLine in trainTextVector:
         textAssemble = textAssemble | set(trainTextVectorLine)
     charactorLabelVector = list(textAssemble)
     return charactorLabelVector

#calculate charactor value vector
#input  charactorLabelVector: List
#       goalTextVector: List
#output charactorValueVector : charactor value vector fof goalTextVector, List
def calculatetCharactorValueVector(charactorLabelVector, targetTextVector):
     def true2one(bool):
         if bool:
             return 1
         else:
             return 0
     charactorValueVector = [ true2one(text in targetTextVector) for text in charactorLabelVector]
     return  charactorValueVector

#classify by bayes
def classifyByBayes(trainCharactorValueVector, classVector, targetCharactorValueVector):
    trainCharactorValueVector = numpy.array(trainCharactorValueVector)
    pClass1 = sum(classVector)/float(len(classVector))
    pClass0 = 1 - pClass1
    numVectorClass1 = numpy.ones(trainCharactorValueVector.shape[1])
    numVectorClass0 = numpy.ones(trainCharactorValueVector.shape[1])
    numClass1 = 2.0
    numClass0 = 2.0
    for i in range(len(classVector)):
        if classVector[i] == 1:
            numVectorClass1 += trainCharactorValueVector[i]
            # numClass1 += 1
            numClass1 += sum(trainCharactorValueVector[i])  #question?
        else:
            numVectorClass0 += trainCharactorValueVector[i]
            # numClass0 += 1
            numClass0 += sum(trainCharactorValueVector[i])
    pVectorClass1 = numVectorClass1/numClass1  # P(xi=1|class1)
    pVectorClass0 = numVectorClass0/numClass0

    # calculate p(x|class1) and p(x|class0)
    pXClass1 = 1
    pXClass0 = 1
    for i in range(len(targetCharactorValueVector)):
        if targetCharactorValueVector[i] == 1:
            pXClass1 = pXClass1*pVectorClass1[i]
            pXClass0 = pXClass0*pVectorClass0[i]
        else:
            pXClass1 = pXClass1*(1 - pVectorClass1[i])
            pXClass0 = pXClass0*(1 - pVectorClass0[i])

    pClass1X = (pXClass1*pClass1)/(pXClass1*pClass1 + pXClass0*pClass0)
    pClass0X = (pXClass0*pClass0)/(pXClass1*pClass1 + pXClass0*pClass0)

    if pClass0X > pClass1X:
        return 0
    else:
        return 1

# translate string to textVector
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>0 ]

# get email text and translate to textVector and classVector
def getTextVector():
    textVector = []; classVector = []
    for i in range(1,26):
        bigString = open('./email/spam/%s.txt'%i).read()
        textVector.append(textParse(bigString))
        classVector.append(1)
        bigString = open('./email/ham/%s.txt'%i).read()
        textVector.append(textParse(bigString))
        classVector.append(0)
    return textVector, classVector

#text NB
def main():
    trainTextVector, trainClassVector = createTextVector()
    trainCharactorLabelVector = getCharactorLabelVector(trainTextVector)
    trainCharactorValueVector = []
    for trainTextLine in trainTextVector:
        trainCharactorValueVector.append(calculatetCharactorValueVector(trainCharactorLabelVector,trainTextLine))
    targetCharactorValueVector = calculatetCharactorValueVector(trainCharactorLabelVector, ['you','are','man'])
    classifyByBayes(trainCharactorValueVector, trainClassVector, targetCharactorValueVector)
    # print trainCharactorValueVector

# classify email by NB
def classifyEmailByNB():
    textVector, classVector  = getTextVector()
    trainTextVector = textVector[0:15]
    trainClassVector = classVector[0:15]
    testTextVector = textVector[15:]
    testClassVector = classVector[15:]

    #translate to Value Vector
    CharactorLabelVector = getCharactorLabelVector(trainTextVector)
    trainCharactorValueVector = []; testCharactorValueVector = []
    for trainTextLine in trainTextVector:
        trainCharactorValueVector.append(calculatetCharactorValueVector(CharactorLabelVector,trainTextLine))
    for testTextLine in testTextVector:
        testCharactorValueVector.append(calculatetCharactorValueVector(CharactorLabelVector,testTextLine))

    #start test
    rightNumber =0; totalNumber = 0;
    for testLine in testCharactorValueVector:
        classifyResult = classifyByBayes(trainCharactorValueVector, trainClassVector, testLine)
        trueResult = testClassVector[testCharactorValueVector.index(testLine)]
        if trueResult == classifyResult:
            rightNumber += 1
        totalNumber += 1
    print "The right rate is", float(rightNumber)/totalNumber



if __name__ == "__main__":
    classifyEmailByNB()