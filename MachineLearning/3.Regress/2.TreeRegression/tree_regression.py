# -*- coding: utf-8 -*-
'''
    this is a model for tree regression
    @author: Liu Weijie
'''
from numpy import *


def load_data(filename):
    data_mat = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data_mat.append([float(x) for x in line.strip('\n').split('\t')])
    return data_mat


def bin_split_dataset(data_set, feature, value):
    data_mat = mat(data_set)
    data_mat1 = data_mat[nonzero(data_mat[:,feature] >= value)[0], :][0]
    data_mat2 = data_mat[nonzero(data_mat[:,feature] < value)[0], :][0]
    return data_mat1, data_mat2


def reg_leaf(data_set):
    return mean(mat(data_set)[:, -1])


def reg_err(data_set):
    data_mat = mat(data_set)
    return var(data_mat[:-1])*shape(data_mat)[0]


def linear_solve(data_set):
    data_mat = mat(data_set)
    m, n = shape(data_mat)
    X = mat(ones((m, n-1))); Y = mat(ones((m, 1)))
    X[:,0:n-1] = data_mat[:,0:n-1]
    Y = data_mat[:, -1]
    XTX = X.T*X
    if linalg.det(XTX) == 0.0:
        raise NameError('this is martix cannot be inverse')
    ws = XTX.I*(X.T*Y)
    return ws, X, Y


def model_leaf(data_set):
    ws, X, Y = linear_solve(data_set)
    return ws


def model_err(data_set):
    ws, X, Y = linear_solve(data_set)
    y_hat = X*ws
    return sum(power(Y - y_hat, 2))


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    data_mat = mat(data_set); tol_S = ops[0]; tol_N = ops[1]
    m, n = shape(data_mat)

    min_error = inf
    for feature in range(n-1):
        for value in set(data_mat[:,feature]):
            data_mat1, data_mat2 = bin_split_dataset(data_mat, feature, value)
            if (shape(data_mat1)[0] < tol_N) or (shape(data_mat2)[0] < tol_N): continue  # 防止某割数据集样本数为0
            error1 = err_type(data_mat1)
            error2 = err_type(data_mat2)
            error_sum = error1 + error2
            if error_sum < min_error:
                split_feature = feature
                split_value = value
                min_error = error_sum

    error_init = err_type(data_mat)
    if (error_init - min_error) <= tol_S:  # 如果分割后误差减小的不是很大, 则不分割
        return None, leaf_type(data_mat)

    data_mat1, data_mat2 = bin_split_dataset(data_mat, split_feature, split_value)
    if (shape(data_mat1)[0] < tol_N) or (shape(data_mat2)[0] < tol_N): # 如果分割后的数据集很小，则不分割
        return None, leaf_type(data_mat)

    return split_feature, split_value[0,0]


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    data_mat = mat(data_set)
    split_feature, split_value = choose_best_split(data_mat, leaf_type, err_type, ops)
    if split_feature is None: return split_value

    ret_tree = {}
    ret_tree['feature'] = split_feature
    ret_tree['value'] = split_value
    right_data_mat, left_data_mat = bin_split_dataset(data_mat, split_feature, split_value)
    ret_tree['right'] = create_tree(right_data_mat, leaf_type, err_type, ops)
    ret_tree['left'] = create_tree(left_data_mat, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']): tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']): tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left'])/2.0


def prune(tree, test_data_set):
    test_data_mat = mat(test_data_set);

    if shape(test_data_mat)[0] == 0: return get_mean(tree)

    if is_tree(tree['right']) or is_tree(tree['left']):
        right_data, left_data = bin_split_dataset(test_data_mat, tree['feature'], tree['value'])

    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right_data)
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], left_data)

    if not is_tree(tree['right']) and not is_tree(tree['left']):
        right_data, left_data = bin_split_dataset(test_data_mat, tree['feature'], tree['value'])
        # 如果误差平方和减小，则合并，否则不和并。
        error_no_merge = sum(power(right_data[:,-1] - tree['right'], 2)) + sum(power(left_data[:,-1] - tree['left'], 2))
        tree_mean = get_mean(tree)
        error_merge = sum(power(test_data_mat[:,-1] - tree_mean, 2))
        if error_merge <= error_no_merge:
            print 'merge'
            return tree_mean
        else:
            return tree
    else:
        return tree


def reg_tree_eval(model, in_data):
    return float(model)


def model_tree_eval(model, in_data):
    return float(model*in_data)


def tree_forecast(tree, in_data, model_eval = reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data[tree['feature']] >= tree['value']:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)
    else:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)


# example use model_tree to regress
def example_model_tree():
    data_set = load_data('ex2.txt')
    model_tree = create_tree(data_set, leaf_type=model_leaf, err_type=model_err, ops=(0,4))
    result = tree_forecast(model_tree, mat([[0.3]]), model_eval=model_tree_eval)
    print 'result by model_tree:',result


# example use reg_tree to regress
def example_reg_tree():
    data_set = load_data('ex2.txt')
    data_set_test = load_data('ex2test.txt')
    reg_tree_init = create_tree(data_set,leaf_type=reg_leaf, err_type=reg_err, ops=(0,4))
    reg_tree = prune(reg_tree_init, data_set_test) # 用测试数据对树进行剪枝
    result = tree_forecast(reg_tree, mat([[0.3]]), model_eval=reg_tree_eval)
    print 'result by reg_tree:',result

if __name__ == '__main__':
    example_reg_tree()
    example_model_tree()






































































