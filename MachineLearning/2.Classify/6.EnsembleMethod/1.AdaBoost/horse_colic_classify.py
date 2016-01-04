'''
this is an example to classify horse colic by adaboost
@author: Liu Weijie
'''
from numpy import *
from adaboost import adaboost_stump, adaboost_stump_classify

def load_data(filename):
    data_list = []; label_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data_list.append([ float(x) for x in line.strip('\n').split("\t")[:-1]])
            label_list.append(float(line.strip('\n').split("\t")[-1]))
    return data_list, label_list


def test_classify():
    train_data_list, train_label_list = load_data("horseColicTraining2.txt")
    stump_list = adaboost_stump(train_data_list, train_label_list, 10)

    test_data_list, test_label_list = load_data("horseColicTest2.txt")
    classify_result = adaboost_stump_classify(test_data_list, stump_list)

    # calculte error rate
    error_array = zeros((1,len(test_data_list)))
    error_array[mat(test_label_list) != mat(classify_result)] = 1.0
    error_rate = sum(error_array)/float(len(test_data_list))
    print "error_rate:", error_rate

if __name__ == '__main__':
    test_classify()



