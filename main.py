# coding = utf-8
# Author: Yu Xin
# Create time: 2024-6-20

import copy
import random
import numpy as np
import pandas as pd
import utils as ut
import seaborn as sns
import matplotlib.pyplot as plt
from model import *


def get_feature(hgw1, hgw2):
    hgw1_temp = hgw1.reshape(360, -1)  # 360*8960
    hgw1_fc = hgw1_temp.sum(1)
    hgw1_fc = abs(hgw1_fc)
    hgw1_temp2 = hgw1.reshape(360, -1, 180)
    w1mulw2 = hgw1_temp2.matmul(hgw2).reshape(360, -1)
    w1mulw2_fc = abs(w1mulw2).sum(1)
    features = w1mulw2_fc
    feat = abs(features) / 10
    a, b, c, d = np.array_split(feat, 4)
    feat = a + b + c + d

    B = np.argsort(feat)
    B = list(reversed(B))  # B中存储排序后的下标
    A = sorted(feat, reverse=True)  # A中存储排序后的结果
    AA = torch.tensor(A)
    BB = torch.tensor(B)
    return AA, BB


def train_model(network, criterion, optimizer, n_epochs=100, print_freq=50):
    train_loss = []
    valid_loss = []
    test_loss = []

    alpha = 0.1
    best_val = 1e10
    best_model_w = copy.deepcopy(network.state_dict())

    for epoch in range(n_epochs):
        network.train()

        # 梯度清零
        optimizer.zero_grad()

        outputs, cor_loss = network(x_train, G_train, y_train)
        # outputs = network(x_train, G_train, y_train)
        loss = criterion(outputs, z_train)
        loss = loss + alpha * cor_loss
        train_loss.append(loss.item())

        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 验证
        with torch.no_grad():
            network.eval()
            outputs, cor_loss = network(x_valid, G_valid, y_valid)
            # outputs = network(x_valid, G_valid, y_valid)
            loss = criterion(outputs, z_valid)
            loss = loss + alpha * cor_loss
            valid_loss.append(loss.item())

            if loss < best_val:
                best_val = loss
                best_model_w = copy.deepcopy(network.state_dict())

        # 加载模型参数
        network.load_state_dict(best_model_w)
        outputs, cor_loss = network(x_test, G_test, y_test)
        # outputs = network(x_test, G_test, y_test)
        loss = criterion(outputs, z_test)
        loss = loss + alpha * cor_loss
        test_loss.append(loss.item())

        if epoch % print_freq == 0:
            print("Epoch {:d}: train_loss: {:.3f} - valid_loss: {:.3f} - test_loss: {:.3f}"
                  .format(epoch + 1, train_loss[epoch], valid_loss[epoch], test_loss[epoch]))

    print("Epoch {:d}: train_loss: {:.3f} - valid_loss: {:.3f} - test_loss: {:.3f}"
          .format(epoch + 1, train_loss[epoch], valid_loss[epoch], test_loss[epoch]))

    return model, outputs, train_loss, valid_loss, test_loss


if __name__ == '__main__':

    # 数据
    algorithms = ['DBO', 'FOA', 'HHO', 'PSO', 'WOA', 'SABO', 'MDBO']
    values = [4.17, 7.00, 3.21, 2.21, 3.93,  5.79, 1.69]
    plt.rcParams['font.family'] = 'Times New Roman'
    # 创建折线图
    plt.plot(algorithms, values, marker='o', linestyle='-', color='b')
    plt.show()

    # 参数设置
    # a1, a2, a3 = 0.2, 0.3, 0.5  # 超图融合系数
    # n_neigs = 20                # KNN邻居数
    # m_prob = 1.0
    #
    # is_prob = False              # 顶点-边矩阵(True)或二进制(False)
    # is_select = True             # 是否对基因进行特征选择
    #
    # n_hiddens = 50
    # n_outputs = 20
    # n_epoch = 1000
    # learning_rate = 1e-3
    # weight_decay = 1e-5
    #
    # # 数据准备
    # x_train, x_valid, x_test, y_train, y_valid, y_test, z_train, z_valid, z_test =\
    #     ut.prepare_data(seed=29135, is_select=is_select)
    #
    # # 构建超图
    # H1 = ut.construct_H(x_train[:, 0, :], n_neigs, is_prob, m_prob)
    # H2 = ut.construct_H(x_train[:, 1, :], n_neigs, is_prob, m_prob)
    # H3 = ut.construct_H(x_train[:, 2, :], n_neigs, is_prob, m_prob)
    #
    # H_train = a1 * H1 + a2 * H2 + a3 * H3
    # G_train = ut.generate_G(H_train, variable_weight=False)
    #
    # H1 = ut.construct_H(x_valid[:, 0, :], n_neigs, is_prob, m_prob)
    # H2 = ut.construct_H(x_valid[:, 1, :], n_neigs, is_prob, m_prob)
    # H3 = ut.construct_H(x_valid[:, 2, :], n_neigs, is_prob, m_prob)
    #
    # H_valid = a1 * H1 + a2 * H2 + a3 * H3
    # G_valid = ut.generate_G(H_valid, variable_weight=False)
    #
    # H1 = ut.construct_H(x_test[:, 0, :], n_neigs, is_prob, m_prob)
    # H2 = ut.construct_H(x_test[:, 1, :], n_neigs, is_prob, m_prob)
    # H3 = ut.construct_H(x_test[:, 2, :], n_neigs, is_prob, m_prob)
    #
    # H_test = a1 * H1 + a2 * H2 + a3 * H3
    # G_test = ut.generate_G(H_test, variable_weight=False)
    #
    # x_train = np.hstack((x_train[:, 0, :], x_train[:, 1, :], x_train[:, 2, :]))
    # x_valid = np.hstack((x_valid[:, 0, :], x_valid[:, 1, :], x_valid[:, 2, :]))
    # x_test = np.hstack((x_test[:, 0, :], x_test[:, 1, :], x_test[:, 2, :]))
    #
    # # 构建网络
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x_train = torch.Tensor(x_train).to(device)
    # x_valid = torch.Tensor(x_valid).to(device)
    # x_test = torch.Tensor(x_test).to(device)
    # G_train = torch.Tensor(G_train).to(device)
    # G_valid = torch.Tensor(G_valid).to(device)
    # G_test = torch.Tensor(G_test).to(device)
    #
    # y_train = torch.Tensor(y_train).to(device)
    # y_valid = torch.Tensor(y_valid).to(device)
    # y_test = torch.Tensor(y_test).to(device)
    #
    # z_train = torch.Tensor(z_train).to(device)
    # z_valid = torch.Tensor(z_valid).to(device)
    # z_test = torch.Tensor(z_test).to(device)
    #
    # # 模型训练
    # model = DSIgnn(x_train.shape[1], y_train.shape[1], n_hiddens, n_outputs)
    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # model, pred, train_loss, valid_loss, test_loss = train_model(model, criterion=torch.nn.MSELoss(),
    #                                                                     optimizer=optimizer, n_epochs=n_epoch)
    #
    # true_value = z_test.cpu().detach().numpy()
    # pred_value = pred.cpu().detach().numpy()
    # ut.evaluate(true_value, pred_value)
    #
    # # hgw1 = model.state_dict()['hgc1.weight']
    # # hgw2 = model.state_dict()['hgc2.weight']
    # # score, idx = get_feature(hgw1, hgw2)
    # # print(score)
    # # print(idx)
    #
    # # 画图
    # plt.figure(num=1)
    # plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.yticks(fontproperties='Times New Roman')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.plot(range(n_epoch), train_loss, label="Train loss")
    # plt.plot(range(n_epoch), valid_loss, label="Valid loss")
    # plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    # plt.show()
    #
    # plt.figure(num=2)
    # plt.xlabel('Subject', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('MMSE', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.yticks(fontproperties='Times New Roman')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.plot(range(z_test.shape[0]), true_value, marker='o', label="True Value")
    # plt.plot(range(z_test.shape[0]), pred_value, marker='*', label="Predicted Value")
    # plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    # plt.show()
    #
    # plt.figure(num=3)
    # residuals = true_value - pred_value
    # plt.xlabel('Predicted Value', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('True Value', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.scatter(pred_value, residuals)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.show()
    #
    # value = np.hstack((true_value, pred_value))
    # value = pd.DataFrame(value, columns=['True', 'Predicted'])
    # sns.lmplot(x='Predicted', y='True', data=value, palette='tab10')
    # plt.xlabel('Predicted Value', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('True Value', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.show()
