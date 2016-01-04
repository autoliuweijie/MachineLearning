# -*- coding: UTF-8 -*-
'''
this module cantains adaboost classify with stumps
@author: Liu Weijie
'''
from numpy import *
import math


def load_simple_data():
    data_mat = mat([
                [1. , 2.1],
                [2. , 1.1],
                [1.3, 1. ],
                [1. , 1. ],
                [2. , 1. ],
    ])
    label_mat = mat([1.0, 1.0, -1.0, -1.0, 1.0])
    return data_mat, label_mat


def stump_classify(data_mat, dimen, thresh_value, tresh_ineq):
    """
    classify simple by stump
    :param data_mat: mat - feature matrix
    :param dimen: int -  the dimen of feature the thresh_value refers to
    :param thresh_value: float - thresh value
    :param tresh_ineq: char - '>' or '<'
    """
    data_mat = mat(data_mat)
    classify_result = mat(ones((shape(data_mat)[0],1)))
    if tresh_ineq == '>':
        classify_result[data_mat[:, dimen] <= thresh_value] = -1.0;
    else:
        classify_result[data_mat[:, dimen] >= thresh_value] = -1.0;
    return classify_result


def train_best_stump(data_mat, label_mat, D):
    min_error_rate = inf; num_steps = 10.0;
    best_dimen = 0; best_thresh_value = 0; best_thresh_ineq = '>';
    m, n = shape(data_mat)
    data_mat = mat(data_mat); label_mat = mat(label_mat).T

    for i in range(n):
        step_gap = (data_mat[:,i].max() - data_mat[:,i].min())/num_steps
        # 注意！！例如给0～1确定分割阈值的时候，阈值的取值不应该只在0～1之间，还应该包括一个比0小的值，和一个比0大的值。
        for j in linspace(data_mat[:,i].min()-step_gap, data_mat[:,i].max()+step_gap, num_steps):
            for ineq in ['>', '<']:
                # calculte error_rate
                classify_result = stump_classify(data_mat, i, j, ineq)
                error_array = zeros((m,1))
                error_array[classify_result != label_mat] = 1.0
                error_rate = D.T*mat(error_array)

                # print "split: dim %d, tresh: %.2f,ineq: %s error_rate: %.3f"%(i, j, ineq, error_rate)

                if error_rate < min_error_rate:
                    min_error_rate = error_rate
                    best_dimen = i
                    best_thresh_value = j
                    best_thresh_ineq = ineq
                    best_classify_result = classify_result

    best_stump = {
        'dimen': best_dimen,
        'thresh_value': best_thresh_value,
        'thresh_ineq': best_thresh_ineq,
        'error_rate': min_error_rate,
        'classify_result': best_classify_result,
    }
    return best_stump


def adaboost_stump(data_list, label_list, num_it):
    """
    get stump_list by adaboost
    :param data_list: list - figure matrix
    :param label_list: list - label matrix in row
    :param num_it: int - max iterate number
    :return: stump_list
    :example     data_mat, label_mat = load_simple_data()
                 print adaboost_stump(data_mat, label_mat, 10)
    """
    data_mat = mat(data_list); label_mat = mat(label_list)
    D = ones((shape(data_mat)[0],1))/float(shape(data_mat)[0])
    stump_list = []
    add_classify_result = mat(zeros((shape(data_mat)[0],1)))
    m, n = shape(data_mat)
    # 迭代num_it次：
    for i in range(num_it):
        # 用D求最优stump
        stump = train_best_stump(data_mat, label_mat, D)
        # 计算alpha
        alpha = float(0.5*math.log((1.0 - stump['error_rate'])/max(stump['error_rate'], 1e-16)))
        stump['alpha'] = alpha
        stump_list.append(stump)
        # 目前迭代的分类的加权累加
        add_classify_result += alpha*stump['classify_result']
        # 计算迭代后的错误率
        error_array = zeros((m,1))
        error_array[sign(add_classify_result) != label_mat.T] = 1.0
        error_rate = sum(error_array)/m

        # print "D:", D.T, "\n"
        # print "classify:", stump['classify_result'].T, "\n"
        # print "add_classify:", add_classify_result.T, "\n"
        # print "error_rate", error_rate, "\n"

        if error_rate <= 0:
            break

        # 更新D
        expon = multiply(-1*alpha*stump['classify_result'], label_mat.T)
        D = multiply(D, exp(expon))
        D = D/D.sum()

    return stump_list


def adaboost_stump_classify(data_to_class, stump_list):
    """
    use stump_list to classify
    :param data_to_class: list - figure list need to be classify like [1,3]
    :param stump_list: list - stump_list returned by adaboost_stump()
    :return: classify result: matrix - result matrix in column
    :example:
        data_mat, label_mat = load_simple_data()
        stump_list = adaboost_stump(data_mat, label_mat, 10)
        print adaboost_stump_classify([5, 5], stump_list)
    """
    data_to_class = mat(data_to_class)
    add_classify_result = mat(zeros((shape(data_to_class)[0],1)))

    for stump in stump_list:
        classify_result = stump_classify(data_to_class, stump['dimen'], stump['thresh_value'], stump['thresh_ineq'])
        add_classify_result += stump['alpha']*classify_result

    return_classify_result = sign(add_classify_result).T.tolist()[0]
    return return_classify_result


if __name__ == '__main__':
    data_mat, label_mat = load_simple_data()
    print label_mat
    stump_list = adaboost_stump(data_mat, label_mat, 10)
    print adaboost_stump_classify([5, 5], stump_list)





























