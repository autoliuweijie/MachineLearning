# -*- coding: UTF-8 -*-
'''
	this module is an example of classifying stealth by decision tree
	@author: Liu Weijie
'''
import trees
import drawATree

#get data
def getData(filename):
    fr = open(filename)
    dataList = [line.strip().split('\t') for line in fr.readlines()]
    labelList = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataList, labelList

if __name__ == '__main__':
    dataSet, labelList = getData('lenses.txt')
    print dataSet
    myTree = trees.createTree(dataSet, labelList)
    drawATree.drawTree(myTree)
    data1 = ['pre','myope','no','reduced']
    print 'this is ', trees.classifyByTree(myTree, labelList, data1)