# -*- coding: UTF-8 -*-
'''
	this module can be used to draw a tree by matlotlib
	@author: Liu Weijie
'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

# get the number of leafs in a tree
def getNumLeafs(myTrees):
	if type(myTrees).__name__ != "dict":
		return 1
	firstKey =myTrees.keys()[0]
	secondList = myTrees[firstKey]
	if type(secondList).__name__ != "dict":
		return 1
	numLeaf = 0  #if numLeaf = 1 ,then it is get number of Node
	for key in secondList:
		numLeaf = numLeaf + getNumLeafs(secondList[key])
	return numLeaf

# get the depth of a Tree
def getDepth(myTrees):
	if type(myTrees).__name__ != "dict":
		return 1
	firstKey =myTrees.keys()[0]
	secondList = myTrees[firstKey]
	if type(secondList).__name__ != "dict":
		return 1
	maxDepth = 0
	for key in secondList:
		thisDepth = getDepth(secondList[key])
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth + 1

# plot a node or leaf
def plotNode(targetAxis, nodeTxt, centerPt, parentPt, nodeType):
	targetAxis.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction', xytext = centerPt, va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

def plotMidText(targetAxis, cntrPt, parentPt, txtStrings):
	xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
	drawTree.ax1.text(xMid, yMid, txtStrings, va="center", ha="center", rotation=30)

# plot a tree in axis
def plotTrees(myTree, parentPt, nodeText):
	numLeaf = getNumLeafs(myTree)
	depth = getDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrPt = (plotTrees.xOff + (1.0 + float(numLeaf))/2.0/plotTrees.totalW, plotTrees.yOff)
	plotMidText(drawTree.ax1 ,cntrPt, parentPt, nodeText)
	plotNode(drawTree.ax1, firstStr, cntrPt, parentPt, decisionNode)
	plotTrees.yOff = plotTrees.yOff - 1.0/plotTrees.totalD
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTrees(secondDict[key], cntrPt, str(key))
		else:
			plotTrees.xOff = plotTrees.xOff + 1.0/plotTrees.totalW
			plotNode(drawTree.ax1, secondDict[key], (plotTrees.xOff, plotTrees.yOff), cntrPt, leafNode)
			plotMidText(drawTree.ax1, (plotTrees.xOff,plotTrees.yOff), cntrPt, str(key) )
	plotTrees.yOff = plotTrees.yOff + 1.0/plotTrees.totalD

# draw a tree
def drawTree(inTree):
	fig = plt.figure(1, facecolor = 'white')
	fig.clf()
	axprops = dict(xticks = [], yticks = [])
	drawTree.ax1 = plt.subplot(111, frameon = False, **axprops)
	plotTrees.totalW = float(getNumLeafs(inTree))
	plotTrees.totalD = float(getDepth(inTree))
	plotTrees.xOff = -0.5/plotTrees.totalW
	plotTrees.yOff = 1.0
	plotTrees(inTree,(0.5, 1.0), '')
	plt.show()

# test
if __name__ == '__main__':
	a = {'asd':{'a':1}}
	b = {'asd':{'a':a, 'b':a, 'c':{"sas":{'a':a}}}}
	drawTree(b)

