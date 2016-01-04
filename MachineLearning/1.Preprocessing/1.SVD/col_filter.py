# coding=utf-8
'''
    This is an recommond system by collavorative filtering improved by SVD
    Author: Liu Weijie
'''
from numpy import *
from numpy.ma import logical_and
import svd

# 四种相似度
def sim_eulid(in_a, in_b):
    return 1.0 / (1.0 + linalg.norm(in_a - in_b))


def sim_person(in_a, in_b):
    if len(in_a) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(in_a, in_b, rowvar=0)[0][1]


def sim_cos(in_a, in_b):
    num = float(in_a.T * in_b)
    denom = linalg.norm(in_a) * linalg.norm(in_b)
    return 0.5 + 0.5 * (num / denom)


# 普通的给data_mat[user, item]打分的办法
def stand_est(data_mat, user, sim_means, item):
    n = shape(data_mat)[1]
    all_sim = 0
    item_score = 0
    for j in range(n):
        score = data_mat[user, j]
        if score == 0:
            continue
        over_lap = nonzero(logical_and(data_mat[:, j] > 0, data_mat[:, item] > 0))[0]
        sim = sim_means(data_mat[over_lap, j], data_mat[over_lap, item])
        all_sim += sim
        item_score += sim * score

    if all_sim == 0:
        item_score = 0
    else:
        item_score = item_score / all_sim

    return item_score


# 用SVD处理过后的data_mat[user, item]打分的办法
def svd_est(data_mat, user, sim_means, item):
    data_mat = svd.svd_low_dimen(data_mat)
    n = shape(data_mat)[1]
    all_sim = 0
    item_score = 0
    for j in range(n):
        score = data_mat[user, j]
        if score == 0:
            continue
        over_lap = nonzero(logical_and(data_mat[:, j] > 0, data_mat[:, item] > 0))[0]
        sim = sim_means(data_mat[over_lap, j], data_mat[over_lap, item])
        all_sim += sim
        item_score += sim * score

    if all_sim == 0:
        item_score = 0
    else:
        item_score = item_score / all_sim

    return item_score


# 推荐函数, 根据data_mat, 把user没评价过的作品中挑选3个推荐
def recommond(data_mat, user, n=3, sim_means=sim_eulid, est_method=stand_est):
    noscore_indx = nonzero(data_mat[user, :].A == 0)[1]
    if len(noscore_indx) == 0:
        return "you tared everything!"
    itemscore = []
    for index in noscore_indx:
        est_score = est_method(data_mat, user, sim_means, index)
        itemscore.append((index, est_score))
    sorted_item = sorted(itemscore, key=lambda one: one[1], reverse=True)[:n]  # 选出前n个推荐
    return sorted_item


if __name__ == '__main__':
    data_set = svd.load_data()
    print 'recommond by stand_est:', \
        recommond(mat(data_set), 2, 2, sim_means=sim_eulid, est_method=stand_est)
    print 'recommond by svd_est:', \
        recommond(mat(data_set), 2, 2, sim_means=sim_eulid, est_method=svd_est)



















