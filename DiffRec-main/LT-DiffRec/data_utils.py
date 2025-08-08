import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
# import torch.sparse as sp
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path, w_min, w_max):
    """
    加载并处理训练、验证和测试数据
    参数:
        train_path: 训练数据路径
        valid_path: 验证数据路径
        test_path: 测试数据路径
        w_min: 最小权重值
        w_max: 最大权重值
    """
    train_list = np.load(train_path, allow_pickle=True)  # 加载训练数据
    valid_list = np.load(valid_path, allow_pickle=True)  # 加载验证数据
    test_list = np.load(test_path, allow_pickle=True)    # 加载测试数据

    uid_max = 0  # 用户ID最大值
    iid_max = 0  # 物品ID最大值
    train_dict = {}  # 用于存储训练数据的字典

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []  # 如果用户不存在，创建新的列表
        train_dict[uid].append(iid)  # 将物品ID添加到用户的交互列表中
        if uid > uid_max:
            uid_max = uid  # 更新最大用户ID
        if iid > iid_max:
            iid_max = iid  # 更新最大物品ID
    
    n_user = uid_max + 1  # 用户总数
    n_item = iid_max + 1  # 物品总数
    print(f'user num: {n_user}')  # 打印用户数量
    print(f'item num: {n_item}')  # 打印物品数量

    train_weight = []  # 存储权重
    train_list = []    # 存储用户-物品对
    for uid in train_dict:
        int_num = len(train_dict[uid])  # 获取用户交互数量
        weight = np.linspace(w_min, w_max, int_num)  # 生成线性递增的权重
        train_weight.extend(weight)  # 添加权重
        for iid in train_dict[uid]:
            train_list.append([uid, iid])  # 添加用户-物品对
    train_list = np.array(train_list)  # 转换为numpy数组

    train_data_temp = sp.csr_matrix((train_weight, \
                (train_list[:, 0], train_list[:, 1])), dtype='float64', \
                shape=(n_user, n_item))

    train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data_temp, train_data_ori, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    """
    自定义数据集类，用于扩散模型
    继承自PyTorch的Dataset类
    """
    def __init__(self, data):
        """
        初始化数据集
        参数:
            data: 输入数据
        """
        self.data = data  # 存储数据

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        参数:
            index: 数据索引
        返回:
            对应的数据项
        """
        item = self.data[index]
        return item

    def __len__(self):
        """
        返回数据集长度
        返回:
            数据集中样本的数量
        """
        return len(self.data)
