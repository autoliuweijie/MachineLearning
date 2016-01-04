# coding=utf-8
'''
    for SVD
'''
from numpy import *


def load_data():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def svd_low_dimen(data_set):
    data_mat = mat(data_set)
    U, sigma, VT = linalg.svd(data_mat)
    # calculte dimen
    all_power = sum(sigma ** 2)
    temp_power = 0
    dimen = 0
    while temp_power < all_power * 0.9:
        temp_power += sigma[dimen] ** 2
        dimen = dimen + 1
    # print "dimen:", dimen

    # low dimen
    sig = mat(eye(dimen) * sigma[:dimen])
    xformed_items = data_mat.T * U[:, :dimen] * sig.I
    return xformed_items.T


def svd_reconstruct(data_set):
    data_mat = mat(data_set)
    U, sigma, VT = linalg.svd(data_mat)
    # calculte dimen
    all_power = sum(sigma ** 2)
    temp_power = 0
    dimen = 0
    while temp_power < all_power * 0.9:
        temp_power += sigma[dimen] ** 2
        dimen = dimen + 1
    # print "dimen:", dimen
    # reconstruct data set
    sig = mat(eye(dimen) * sigma[:dimen])
    ret_data_set = U[:, :dimen] * sig * VT[:dimen, :]
    return ret_data_set


if __name__ == '__main__':
    old_data = mat(load_data())
    low_data = svd_low_dimen(old_data)
    rec_data = svd_reconstruct(old_data)

























