'''
    for PCA
'''
from numpy import *


def load_data(filename, split_mark='\t'):
    data_set = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_temp = [float(v) for v in line.strip().split(split_mark)]
            data_set.append(line_temp)
    return data_set


def pca(data_set, n=999999):
    data_mat = mat(data_set)
    mean_vals = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals

    cov_mat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vector = linalg.eig(mat(cov_mat))
    # get the top n eig_vector
    eig_val_index = argsort(eig_vals)[:-(n + 1):-1]
    red_eig_vector = eig_vector[:, eig_val_index]
    # translate to the new axis
    new_feat = mean_removed * red_eig_vector

    return new_feat, red_eig_vector


if __name__ == '__main__':
    data_set = load_data('testSet.txt')
    print pca(data_set, 1)
