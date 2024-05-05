import torch
import torch.nn.functional as F
import torch.nn as nn


def adversary_loss(t_joint,t_shuffled,x_real,CONTRAST_WEIGHT):
    # Softplus函数可以看作是ReLU函数的平滑，一直为正数
    Ej = -F.softplus(-t_joint)
    Em = F.softplus(t_shuffled)
    # shape[0]就是第一维度有多少行
    GLOBAL = torch.sum(Em - Ej) / x_real.shape[0] * CONTRAST_WEIGHT
    return GLOBAL

def adversary_kl_loss(term_a, term_b, x_real, z_inferred):
    # # 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充。
    # prior = torch.randn_like(z_inferred)
    # # tensor:64
    # term_a = torch.log(torch.sigmoid(
    #     adversary_prior(x_real, z_inferred) + 1e-9))
    # # tensor:64 padding:tensor64 fasle,fasle,.....true,true...
    # term_b = torch.log(1.0 - torch.sigmoid(adversary_prior(x_real,prior)) + 1e-9)
    PRIOR = -torch.mean(term_a + term_b, dim=-1)
    PRIOR = torch.sum(PRIOR) / x_real.shape[0]

    return PRIOR

class Adversary(nn.Module):
    def __init__(self, embedding_dim):
        # 判别网络
        super(Adversary, self).__init__()
        self.embedding_dim = embedding_dim
        # 128,64
        self.linear_i = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        self.dnet_list = []
        self.net_list = []
        for _ in range(2):
            self.dnet_list.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            self.net_list.append(nn.Linear(self.embedding_dim, self.embedding_dim))

        self.dnet_list = nn.ModuleList(self.dnet_list)
        self.net_list = nn.ModuleList(self.net_list)

        self.linear_o = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout1 = nn.Dropout(1)
        self.dropout2 = nn.Dropout(1)



    def forward(self, x, z):
        # batch_size x seq_len x dim
        # x:tensor 64 z:tensor 64
        net = self.linear_i(torch.cat([x, z], 1)).to()
        net = self.dropout1(net)

        for i in range(2):
            dnet = self.dnet_list[i](net)
            net = net + self.net_list[i](dnet)
            # ELU函数是针对ReLU函数的一个改进型，相比于ReLU函数，在输入为负数的情况下，是有一定的输出的，
            # 而且这部分输出还具有一定的抗干扰能力。这样可以消除ReLU死掉的问题，
            net = F.elu(net)

        # seq_len
        net = self.linear_o(net)
        net = self.dropout2(net)
        # torch.square z里面每一项都平方
        net = net + 0.5 * torch.square(z)

        net = net

        return net