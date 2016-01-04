# -*-coding: utf-8 -*-
'''
this model is aboult regression
@author: liuweijie
'''
from numpy import *
import matplotlib.pyplot as plt
import math


def load_data(filename):
    data_list = []; label_list =[]
    with open(filename, 'r') as f:
        for line in f.readlines():
             data_list.append([ float(x) for x in line.strip('\n').split("\t")[:-1]])
             label_list.append(float(line.strip('\n').split("\t")[-1]))
    return data_list, label_list


def stand_regress(data_list, label_list):
    """
    stand regress
    :param data_list: list- feature matrix in row
    :param label_list: list- label in row
    :return: list- w vector
    example:
        data_list, label_list = load_data("ex0.txt")
        w = stand_regress(data_list, label_list)
        plot_regress(data_list, label_list, w)
    """
    X = mat(data_list); Y = mat(label_list)
    XTX = X.T*X
    if linalg.det(XTX) == 0.0:
        print "this matrix is singular, cannot to inverse!"
        return None
    else:
        w_estimate = (X.T*X).I * X.T * Y.T
    return w_estimate.T.tolist()[0]


def locally_weighted_linear_regress(test_point, data_list, label_list, k=1.0):
    X_mat = mat(data_list); Y_mat = mat(label_list); test_point = mat(test_point)

    # create weighted_mat
    m = shape(X_mat)[0]
    weighted_mat = mat(eye(m))
    for i in range(m):
        dis = X_mat[i] - test_point
        diff = math.sqrt((dis * dis.T)[0,0])
        weighted_mat[i,i] = exp(diff/(-2.0*k**2))

    # calculate w
    XTX = X_mat.T * (weighted_mat * X_mat)
    if linalg.det(XTX) == 0.0:
        print "this matrix is singular, cannot to inverse!"
        return None
    else:
        w_estimate = (XTX).I * X_mat.T * weighted_mat * Y_mat.T

    # return regress resultt
    label_estimate = (test_point * w_estimate)[0,0]
    return label_estimate


def test_locally_weighted_linear_regress(test_list, data_list, label_list, k=1.0):
    result_list = []
    for i in range(len(test_list)):
        label_estimate = locally_weighted_linear_regress(test_list[i], data_list, label_list, k)
        print "==>", label_estimate, ':', label_list[i]
        result_list.append(label_estimate)
    return result_list

# plot X[:1] and Y
def plot_regress(data_list, label_list, w_estimate):
    X_mat = mat(data_list); Y_mat = mat(label_list); w_estimate = mat(w_estimate)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_mat[:,1].flatten().A[0], Y_mat.flatten().A[0])

    Y_estimate_mat = X_mat * w_estimate
    ax.plot(X_mat[:,1].flatten().A[0], Y_estimate_mat.flatten().A[0])
    plt.show()

    # 求相关系数
    print "corroce:", corrcoef(Y_mat, Y_estimate_mat.T)


def ridge_regress(data_list, label_list, lam=0.2):
    """
    riege_regress
    :param data_list: list- feature matrix in row
    :param label_list: list- label in row
    :param lam: float- lamda default 0.2
    :return: list- w vector
    """
    X = mat(data_list); Y = mat(label_list)
    XTX = X.T*X
    denom = XTX + eye(shape(X)[1])*lam
    if linalg.det(denom) == 0.0:
        print "this matrix is singular, cannot to inverse!"
        return None
    else:
        w_estimate = denom.I * X.T * Y.T

    return w_estimate.T.tolist()[0]


from scipy import stats
def test_ridge_regeress(data_list, label_list):
    X_mat = mat(data_list); Y_mat = mat(label_list)

    # standlize data
    X_mat = stats.zscore(X_mat, 0)
    X_mat[isnan(X_mat)] = 0.0
    Y_mat = Y_mat - mean(Y_mat,1)

    # test regress
    w_list = []
    num_test = 100
    for i in range(num_test):
        w_list.append(ridge_regress(X_mat, Y_mat, exp(i-30)))

    # plot w_list
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w_list)
    plt.show()
    return w_list


def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def stage_regress(data_list, label_list, num_it=1000, eps=0.1):
    """
    stage_regress
    :param data_list: list- figure_matrix
    :param label_list: list- label_matrix in row
    :param num_it: number od itergrate
    :param eps: float- gap
    :return: list- w matrix
    example:
        data_list, label_list = load_data("abalone.txt")
        w_list = stage_regress(data_list, label_list)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w_list)
        plt.show()
    """
    xMat = mat(data_list); yMat=mat(label_list).T

    m,n=shape(xMat)
    returnMat = mat(zeros((num_it, n)))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(num_it):
        # print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat.tolist()


if __name__ == "__main__":
    pass