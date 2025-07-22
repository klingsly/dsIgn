# coding = utf-8
# Author: Yu Xin
# Create time: 2024-6-21

import os
import math
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data(seed, is_select=False):
    # 加载数据
    if not os.path.exists('../Result/image.npy'):
        bl = np.array(pd.read_excel('../Result/Baseline_Vgm.xlsx', index_col=0))
        m06 = np.array(pd.read_excel('../Result/M06_Vgm.xlsx', index_col=0))
        m12 = np.array(pd.read_excel('../Result/M12_Vgm.xlsx', index_col=0))

        bl = np.reshape(bl[:, 1:], (bl.shape[0], 1, bl.shape[1] - 1))
        m06 = np.reshape(m06[:, 1:], (m06.shape[0], 1, m06.shape[1] - 1))
        m12 = np.reshape(m12[:, 1:], (m12.shape[0], 1, m12.shape[1] - 1))

        image = np.concatenate((bl, m06, m12), axis=1)  # (被试数, 时间点, 特征数)
        np.save('../Result/image.npy', image)
    else:
        image = np.array(np.load('../Result/image.npy', allow_pickle=True))

    if not os.path.exists('../Result/gene.npy'):
        gene = np.array(pd.read_csv('../Result/Gene.csv', index_col=0))
        gene = gene.T
        np.save('../Result/gene.npy', gene)
    else:
        gene = np.array(np.load('../Result/gene.npy', allow_pickle=True))

    score = pd.read_excel('../Result/M24.xlsx', index_col=0)
    score = np.array(score[['MMSE']])

    idx = np.arange(0, image.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=seed, shuffle=True)
    idx_valid, idx_test = train_test_split(idx_test, test_size=0.5, random_state=seed, shuffle=True)

    x_train = image[idx_train, :, :]
    x_valid = image[idx_valid, :, :]
    x_test = image[idx_test, :, :]

    # 数据归一化
    scaler = MinMaxScaler()
    x_train[:, 0, :] = scaler.fit_transform(x_train[:, 0, :])
    x_valid[:, 0, :] = scaler.transform(x_valid[:, 0, :])
    x_test[:, 0, :] = scaler.transform(x_test[:, 0, :])

    scaler = MinMaxScaler()
    x_train[:, 1, :] = scaler.fit_transform(x_train[:, 1, :])
    x_valid[:, 1, :] = scaler.transform(x_valid[:, 1, :])
    x_test[:, 1, :] = scaler.transform(x_test[:, 1, :])

    scaler = MinMaxScaler()
    x_train[:, 2, :] = scaler.fit_transform(x_train[:, 2, :])
    x_valid[:, 2, :] = scaler.transform(x_valid[:, 2, :])
    x_test[:, 2, :] = scaler.transform(x_test[:, 2, :])

    scaler = MinMaxScaler()
    gene = scaler.fit_transform(gene)
    y_train = gene[idx_train, :]
    y_valid = gene[idx_valid, :]
    y_test = gene[idx_test, :]

    z_train = score[idx_train]
    z_valid = score[idx_valid]
    z_test = score[idx_test]

    if is_select:
        selector = SelectKBest(chi2, k=200)
        y_train = selector.fit_transform(y_train, z_train.ravel())
        y_valid = selector.transform(y_valid)
        y_test = selector.transform(y_test)

    return x_train, x_valid, x_test, y_train, y_valid, y_test, z_train, z_valid, z_test


def cos_dis(x):
    """
    计算余弦距离，取值范围为[−1,1]，更加注重两个向量在方向上的差异，而非距离或长度上
    :param x: (n_objs, n_feas) 特征矩阵
    :return: (n_objs, n_objs) 距离矩阵
    """
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * (np.linalg.norm(x[j])))
            dist_mat[i][j] = d
    return dist_mat


def eu_dis(x):
    """
    计算欧式距离
    :param x: (n_objs, n_feas) 特征矩阵
    :return: (n_objs, n_objs) 距离矩阵
    """
    x = np.mat(x).astype('float')
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def man_dis(x):
    """
    计算曼哈顿距离
    :param x: (n_objs, n_feas) 特征矩阵
    :return: (n_objs, n_objs) 距离矩阵
    """
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            dist_mat[i][j] = np.sum(np.abs(x[i]-x[j]))
    return dist_mat


def cor_dis(x):
    """
    计算相关系数
    :param x: (n_objs, n_feas) 特征矩阵
    :return: (n_objs, n_objs) 距离矩阵
    """
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = 1 - abs(np.corrcoef(x[i], x[j]))
            dist_mat[i][j] = d[0][1]
    return dist_mat


def evaluate(true_value, pred_value):
    MAE = mean_absolute_error(true_value, pred_value)
    MAPE = mape(true_value, pred_value)
    RMSE = math.sqrt(mean_squared_error(true_value, pred_value))
    R, a = scipy.stats.pearsonr(pred_value, true_value)
    R2 = r2_score(true_value, pred_value)
    print('\n MAE |MAPE | RMSE | R  | R2\n{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'
          .format(MAE, MAPE, RMSE, R[0], R2))


def mape(true_value, pred_value):
    """
    计算MAPE指标
    :param true_value: 真实标签
    :param pred_value: 预测标签
    :return: MAPE指标
    """
    record = []
    for index in range(len(true_value)):
        temp_mape = np.abs((pred_value[index] - true_value[index]) / true_value[index])
        record.append(temp_mape)
    return np.mean(record)


def construct_H(x, n_neigs, is_prob=True, m_prob=1):
    """
    构建超图H
    :param x: (n_objs, n_feas) 特征矩阵
    :param n_neigs: 邻居数
    :param is_prob: 顶点-边矩阵(True)或二进制(False)
    :param m_prob:
    :return: (n_objs, n_edges) 超图H
    """
    dist_mat = eu_dis(x)
    n_objs = dist_mat.shape[0]

    # 从每个节点的中心特征空间构建超边
    H = np.zeros((n_objs, n_objs))

    for i in range(n_objs):
        dist_mat[i, i] = 0
        dist_vec = dist_mat[i]

        # 将dist_vec中的元素从小到大排列，提取其对应的index(索引)，squeeze()，假如某一维只有一项数据，则删除这一维度。
        nearest_idx = np.array(np.argsort(dist_vec)).squeeze()
        dist_avg = np.average(dist_vec)

        if not np.any(nearest_idx[:n_neigs] == i):
            nearest_idx[n_neigs-1] = i

        for j in nearest_idx[:n_neigs]:
            if is_prob:
                H[j, i] = np.exp(-dist_vec[0, j] ** 2 / (m_prob * dist_avg) ** 2)
            else:
                H[j, i] = 1.0
    return H


def generate_G(H, variable_weight=False):
    """
    根据图关联矩阵H计算G
    :param H: (n_objs, n_edges) 超图H
    :param variable_weight: 超边的权重是否可变
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)         # 超边的权重
    DV = np.sum(H * W, axis=1)  # 节点的度
    DE = np.sum(H, axis=0)      # 超边的度

    invDE = np.mat(np.diag(np.power(DE, -1)))   # DE^-1
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))   # DV^-0.5
    W = np.mat(np.diag(W))      # 以W为对角线的矩阵
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
