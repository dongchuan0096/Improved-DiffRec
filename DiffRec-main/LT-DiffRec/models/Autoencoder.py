"""
基于高斯扩散的大规模推荐系统自编码器模型

主要作用：
1. 数据压缩
   - 将用户-物品交互矩阵(n_user × n_item)压缩到低维潜在空间
   - 例如：1000×1000的交互矩阵 → 300维的潜在向量
   - 压缩比：1000×1000/300 ≈ 3333倍

2. 特征提取
   - 学习用户兴趣的潜在表示
   - 捕捉物品之间的相似性
   - 例如：用户A的交互向量[1,0,1,0,1] → 潜在向量[0.2,0.5,0.3]

3. 生成推荐
   - 从潜在空间重建用户-物品交互
   - 预测用户可能感兴趣的物品
   - 例如：潜在向量[0.2,0.5,0.3] → 预测交互[0.8,0.1,0.9,0.2,0.7]

4. 多类别处理
   - 对物品进行聚类
   - 为每个类别构建独立的编码器和解码器
   - 例如：将1000个物品分为3类，每类独立处理

压缩过程示例：
1. 输入数据：
   用户-物品交互矩阵 (1000×1000)
   [
       [1, 0, 1, 0, 1],  # 用户1的交互
       [0, 1, 0, 1, 0],  # 用户2的交互
       ...
   ]

2. 编码过程：
   输入层(1000) → 隐藏层(300) → 潜在层(150)
   - 第一层：1000 → 300 (降维)
   - 第二层：300 → 150 (特征提取)
   - 输出：150维潜在向量

3. 潜在表示：
   [
       [0.2, 0.5, 0.3],  # 用户1的潜在表示
       [0.4, 0.1, 0.6],  # 用户2的潜在表示
       ...
   ]

4. 解码过程：
   潜在层(150) → 隐藏层(300) → 输出层(1000)
   - 第一层：150 → 300 (升维)
   - 第二层：300 → 1000 (重建)
   - 输出：重建的交互矩阵

5. 输出数据：
   重建的交互矩阵 (1000×1000)
   [
       [0.8, 0.1, 0.9, 0.2, 0.7],  # 用户1的预测交互
       [0.2, 0.8, 0.3, 0.7, 0.4],  # 用户2的预测交互
       ...
   ]

优势：
1. 降维：处理大规模数据
2. 特征提取：学习数据本质
3. 生成能力：产生新的推荐
4. 可解释性：潜在空间有语义
5. 灵活性：支持多类别处理

在长尾推荐中的应用：
1. 通过压缩学习物品关系
2. 利用潜在空间生成推荐
3. 平衡热门和长尾物品
4. 提高推荐多样性
"""

import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口
import torch  # 导入PyTorch主库
import numpy as np  # 导入NumPy库
import math  # 导入数学函数库
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_  # 导入权重初始化函数
from kmeans_pytorch import kmeans  # 导入PyTorch版本的K-means聚类算法


class AutoEncoder(nn.Module):
    """
    基于高斯扩散的大规模推荐系统自编码器模型
    """
    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1):
        super(AutoEncoder, self).__init__()  # 调用父类初始化

        self.item_emb = item_emb  # 物品嵌入向量
        self.n_cate = n_cate  # 类别数量
        self.in_dims = in_dims  # 输入维度
        self.out_dims = out_dims  # 输出维度
        self.act_func = act_func  # 激活函数类型
        self.n_item = len(item_emb)  # 物品总数
        self.reparam = reparam  # 是否使用重参数化
        self.dropout = nn.Dropout(dropout)  # Dropout层

        if n_cate == 1:  # 单类别模式（无聚类）
            # 构建编码器维度列表
            in_dims_temp = [self.n_item] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
            # 构建解码器维度列表
            out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item]

            # 构建编码器模块
            encoder_modules = []
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                encoder_modules.append(nn.Linear(d_in, d_out))  # 添加线性层
                if self.act_func == 'relu':
                    encoder_modules.append(nn.ReLU())  # ReLU激活函数
                elif self.act_func == 'sigmoid':
                    encoder_modules.append(nn.Sigmoid())  # Sigmoid激活函数
                elif self.act_func == 'tanh':
                    encoder_modules.append(nn.Tanh())  # Tanh激活函数
                else:
                    raise ValueError
            self.encoder = nn.Sequential(*encoder_modules)  # 创建编码器序列

            # 构建解码器模块
            decoder_modules = []
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                decoder_modules.append(nn.Linear(d_in, d_out))  # 添加线性层
                if self.act_func == 'relu':
                    decoder_modules.append(nn.ReLU())  # ReLU激活函数
                elif self.act_func == 'sigmoid':
                    decoder_modules.append(nn.Sigmoid())  # Sigmoid激活函数
                elif self.act_func == 'tanh':
                    decoder_modules.append(nn.Tanh())  # Tanh激活函数
                elif self.act_func == 'leaky_relu':
                    encoder_modules.append(nn.LeakyReLU())  # LeakyReLU激活函数
                else:
                    raise ValueError
            decoder_modules.pop()  # 移除最后一个激活函数
            self.decoder = nn.Sequential(*decoder_modules)  # 创建解码器序列
        
        else:  # 多类别模式（使用聚类）
            # 使用K-means进行物品聚类
            self.cluster_ids, _ = kmeans(X=item_emb, num_clusters=n_cate, distance='euclidean', device=device)
            # 获取每个类别的物品索引
            category_idx = []
            for i in range(n_cate):
                idx = np.argwhere(self.cluster_ids.numpy() == i).squeeze().tolist()
                category_idx.append(torch.tensor(idx, dtype=int))
            self.category_idx = category_idx  # 存储每个类别的物品索引
            self.category_map = torch.cat(tuple(category_idx), dim=-1)  # 类别映射
            self.category_len = [len(self.category_idx[i]) for i in range(n_cate)]  # 每个类别的物品数量
            print("category length: ", self.category_len)
            assert sum(self.category_len) == self.n_item  # 确保所有物品都被分配

            # 构建编码器和解码器
            encoder_modules = [[] for _ in range(n_cate)]  # 为每个类别创建编码器模块列表
            decode_dim = []  # 存储每个类别的解码维度
            for i in range(n_cate):
                if i == n_cate - 1:
                    # 最后一个类别使用剩余维度
                    latent_dims = list(self.in_dims - np.array(decode_dim).sum(axis=0))
                else:
                    # 根据类别大小分配维度
                    latent_dims = [int(self.category_len[i] / self.n_item * self.in_dims[j]) for j in range(len(self.in_dims))]
                    latent_dims = [latent_dims[j] if latent_dims[j] != 0 else 1 for j in range(len(self.in_dims))]
                in_dims_temp = [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
                decode_dim.append(latent_dims)
                # 构建每个类别的编码器
                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                    encoder_modules[i].append(nn.Linear(d_in, d_out))
                    if self.act_func == 'relu':
                        encoder_modules[i].append(nn.ReLU())
                    elif self.act_func == 'sigmoid':
                        encoder_modules[i].append(nn.Sigmoid())
                    elif self.act_func == 'tanh':
                        encoder_modules[i].append(nn.Tanh())
                    elif self.act_func == 'leaky_relu':
                        encoder_modules[i].append(nn.LeakyReLU())
                    else:
                        raise ValueError

            self.encoder = nn.ModuleList([nn.Sequential(*encoder_modules[i]) for i in range(n_cate)])
            print("Latent dims of each category: ", decode_dim)

            self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]  # 反转解码维度

            if len(out_dims) == 0:  # 单层解码器
                out_dim = self.in_dims[-1]
                decoder_modules = []
                decoder_modules.append(nn.Linear(out_dim, self.n_item))
                self.decoder = nn.Sequential(*decoder_modules)
            else:  # 多层解码器
                decoder_modules = [[] for _ in range(n_cate)]
                for i in range(n_cate):
                    out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
                    for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                        decoder_modules[i].append(nn.Linear(d_in, d_out))
                        if self.act_func == 'relu':
                            decoder_modules[i].append(nn.ReLU())
                        elif self.act_func == 'sigmoid':
                            decoder_modules[i].append(nn.Sigmoid())
                        elif self.act_func == 'tanh':
                            decoder_modules[i].append(nn.Tanh())
                        elif self.act_func == 'leaky_relu':
                            encoder_modules[i].append(nn.LeakyReLU())
                        else:
                            raise ValueError
                    decoder_modules[i].pop()
                self.decoder = nn.ModuleList([nn.Sequential(*decoder_modules[i]) for i in range(n_cate)])
            
        self.apply(xavier_normal_initialization)  # 使用Xavier正态分布初始化权重
        
    def Encode(self, batch):
        batch = self.dropout(batch)  # 应用dropout
        if self.n_cate == 1:  # 单类别编码
            hidden = self.encoder(batch)  # 通过编码器
            mu = hidden[:, :self.in_dims[-1]]  # 均值
            logvar = hidden[:, self.in_dims[-1]:]  # 对数方差

            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)  # 重参数化
            else:
                latent = mu
            
            # 计算KL散度
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            return batch, latent, kl_divergence

        else:  # 多类别编码
            batch_cate = []
            for i in range(self.n_cate):
                batch_cate.append(batch[:, self.category_idx[i]])
            # 分别对每个类别进行编码
            latent_mu = []
            latent_logvar = []
            for i in range(self.n_cate):
                hidden = self.encoder[i](batch_cate[i])
                latent_mu.append(hidden[:, :self.decode_dim[i][0]])
                latent_logvar.append(hidden[:, self.decode_dim[i][0]:])

            mu = torch.cat(tuple(latent_mu), dim=-1)  # 连接所有类别的均值
            logvar = torch.cat(tuple(latent_logvar), dim=-1)  # 连接所有类别的对数方差
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)  # 重参数化
            else:
                latent = mu

            # 计算KL散度
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence
    
    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成随机噪声
        return eps.mul(std).add_(mu)  # 重参数化技巧
    
    def Decode(self, batch):
        if len(self.out_dims) == 0 or self.n_cate == 1:  # 单层解码器
            return self.decoder(batch)
        else:  # 多层解码器
            batch_cate = []
            start=0
            for i in range(self.n_cate):
                end = start + self.decode_dim[i][0]
                batch_cate.append(batch[:, start:end])
                start = end
            pred_cate = []
            for i in range(self.n_cate):
                pred_cate.append(self.decoder[i](batch_cate[i]))
            pred = torch.cat(tuple(pred_cate), dim=-1)  # 连接所有类别的预测结果

            return pred
    
def compute_loss(recon_x, x):
    # 计算多项式对数似然损失
    return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))


def xavier_normal_initialization(module):
    """
    使用Xavier正态分布初始化网络参数
    对于nn.Linear层的偏置项使用常数0初始化
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)  # 使用Xavier正态分布初始化权重
        if module.bias is not None:
            constant_(module.bias.data, 0)  # 使用常数0初始化偏置
                