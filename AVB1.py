# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:27:48 2019

@author: 86187
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from numpy.random import RandomState


    
    
class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
                     nn.Linear(in_features, out_features),nonlinearity)
    def forward(self, x):
        return self.model(x)
    
class DBTKGE(nn.Module):
    def __init__(self, kg, embedding_dim, batch_size, learning_rate, L, gran,thr, gamma, n_day, gpu=True):
        super(DBTKGE, self).__init__()
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_day = n_day
        self.gran = gran
        self.thr = thr

        self.L = L


        # self.emb_T_img.weight.data.uniform_(-r, r)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.sigma_parameter = 1e-3
        self.sample_num=5
        self.esp=1e-10
        self.kl_parameter=1e-3
        self.hidden_dim = self.embedding_dim

        self.s_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.s_mean = LinearUnit(self.hidden_dim * 2, self.embedding_dim)
        self.s_logvar = LinearUnit(self.hidden_dim * 2, self.embedding_dim)


        self.rel_embedding = nn.Embedding(self.kg.n_relation * 2, self.embedding_dim, padding_idx=0)
        self.time_embdding = nn.Embedding(n_day+1, self.embedding_dim)
        self.time2mean_ent = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.time2std_ent = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.time2std_ent = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.time2std_ent = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        self.sd_h = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.sd_t = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        self.ent_mean = nn.Embedding(self.kg.n_entity, self.embedding_dim,_weight=torch.ones(self.kg.n_entity, self.embedding_dim)) #ones ??
        self.ent_std = nn.Embedding(self.kg.n_entity, self.embedding_dim,_weight=torch.zeros(self.kg.n_entity, self.embedding_dim))


        self.transfer_linear_ent = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.transfer_mean_ent = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.transfer_std_ent = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.rel_embedding.weight.data.uniform_(-r, r)
        self.time_embdding.weight.data.uniform_(-r, r)
        self.embed_dropout = nn.Dropout(0)
        self.linear1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.activation = nn.Tanh()
        self.eps = AddEps(self.embedding_dim)
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


        self.device = torch.device('cpu')
        if self.gpu:
            self.cuda()
            self.device = torch.device('cuda')


    def forward_kl_loss(self, pos):

        h, r, t, x_s,mean_std,mean_std1 =self.get_embedding(pos)
        head_mean, head_std, head_mean_pri, head_std_pri, tail_mean, tail_std, tail_mean_pri, tail_std_pri=mean_std
        x_s_mean1,x_s_std1= mean_std1

        # 转移损失(KL损失) -->
        # 转移概率 loss current_mean, current_std, prior_mean, prior_std
        head_trans_loss = self.transfer_kl_loss(head_mean, head_std, head_mean_pri, head_std_pri, False, 'ent')
        tail_trans_loss = self.transfer_kl_loss(tail_mean, tail_std, tail_mean_pri, tail_std_pri, False, 'ent')
        trans_loss=head_trans_loss+tail_trans_loss

        # 静态kl损失
        kl_s = -0.5 * torch.sum(1 + x_s_std1 - torch.pow(x_s_mean1, 2) - torch.exp(x_s_std1))
        # score_s = (torch.sum(torch.abs(h + r - t), 1)).mean()
        return (trans_loss + kl_s )*self.kl_parameter

    def get_embedding(self,X,i=0):
        h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.int64)//self.gran

        if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            r_i = Variable(torch.from_numpy(r_i).cuda())
            d_i = Variable(torch.from_numpy(d_i).cuda())
        else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            r_i = Variable(torch.from_numpy(r_i))
            d_i = Variable(torch.from_numpy(d_i))

        #time
        time_idx=d_i+torch.tensor(1).to(self.device)
        pri_time_laten = self.time_embdding(d_i)
        time_laten = self.time_embdding(time_idx)  # 下一个时间

        #head
        head_mean1 = self.ent_mean(h_i)  # (batch, out_size)

        head_mean_pri = self.time2mean_ent(torch.cat([head_mean1, pri_time_laten], 1))
        head_mean = self.time2mean_ent(torch.cat([head_mean1, time_laten], 1))


        head_std = self.ent_std(h_i)  # (batch, out_size)
        head_std_pri = self.time2std_ent(torch.cat([head_std, pri_time_laten], 1)).mul(0.5).exp()
        head_std = self.time2std_ent(torch.cat([head_std, time_laten], 1)).mul(0.5).exp()

        #tail
        tail_mean1 = self.ent_mean(t_i)  # (batch, out_size)
        tail_mean_pri = self.time2mean_ent(torch.cat([tail_mean1, pri_time_laten], 1))
        tail_mean = self.time2mean_ent(torch.cat([tail_mean1, time_laten], 1))

        tail_std = self.ent_std(t_i)  # (batch, out_size)
        tail_std_pri = self.time2std_ent(torch.cat([tail_std, pri_time_laten], 1)).mul(0.5).exp()
        tail_std = self.time2std_ent(torch.cat([tail_std, time_laten], 1)).mul(0.5).exp()

        r = self.rel_embedding(r_i)
        if i == 2:
            h = self.reparameter(head_mean, head_std)
            t = self.reparameter(tail_mean, tail_std)
            x_s = 0
            # relation
            r = self.rel_embedding(r_i)
            return h, r, t, x_s, (
            head_mean, head_std, head_mean_pri, head_std_pri, tail_mean, tail_std, tail_mean_pri, tail_std_pri) \
                , ( 0, 0)

        if i == 0:
            x_s = head_mean + r - tail_mean
            x_s = x_s.reshape(self.batch_size, -1, self.embedding_dim)
            lstm_out_x_s, _ = self.s_lstm(x_s)
            backward_x_s = lstm_out_x_s[:, 0, self.hidden_dim:self.hidden_dim * 2]
            frontal_x_s = lstm_out_x_s[:, 0, 0:self.hidden_dim]
            lstm_out_x_s = torch.cat((frontal_x_s, backward_x_s), dim=1)
            x_s_mean = self.s_mean(lstm_out_x_s)
            x_s_std = self.s_logvar(lstm_out_x_s)
            x_s = self.reparameter(x_s_mean, x_s_std)


        else:
            x_s = head_mean + r - tail_mean
            x_s = x_s.reshape(self.batch_size, -1, self.embedding_dim)
            lstm_out_x_s, _ = self.s_lstm(x_s)
            backward_x_s = lstm_out_x_s[:, 0, self.hidden_dim:self.hidden_dim * 2]
            frontal_x_s = lstm_out_x_s[:,149, 0:self.hidden_dim]
            lstm_out_x_s = torch.cat((frontal_x_s, backward_x_s), dim=1)
            x_s_mean = self.s_mean(lstm_out_x_s)
            x_s_std = self.s_logvar(lstm_out_x_s)
            x_s = self.reparameter(x_s_mean, x_s_std)

        '''
        隐变量采样
        '''
        # s_mean1, s_logvar1, s1 = self.encode_s(head_mean1)
        h = self.reparameter(head_mean, head_std)
        # s_mean1, s_logvar1, s1 = self.encode_s(tail_mean1)
        t = self.reparameter(tail_mean, tail_std)
        #relation

        return h,r,t,x_s,(head_mean, head_std, head_mean_pri, head_std_pri,tail_mean, tail_std, tail_mean_pri, tail_std_pri)\
               ,(x_s_mean,x_s_std)
    def forward(self, X,i = 0):
        h,r,t,x_s,_,_=self.get_embedding(X,i)
        # 潜在变量
        z_h = self.eps(self.embed_dropout(h))
        z_t = self.eps(self.embed_dropout(t))
        # 解码器
        h = z_h
        t = z_t
        # h = self.linear1(z_h)
        # h = self.activation(h)
        # h = self.linear2(h)
        # t = self.linear1(z_t)
        # t = self.activation(t)
        # t = self.linear2(t)
        if self.L == 'L1':
            score = torch.sum(torch.abs(h + r - t), 1)
            # if i != 2:
            #     score1 = torch.sum(torch.abs(x_s), 1)
            #     score1 = score1.mean()
            # return score,score1
        else:
            score = torch.sum((h + r - t) ** 2, 1)
            score = torch.sqrt(score)
        # score1 = 0
        return score

    def Encoder(self, X, i=0):
        h,r,t,x_s,_,_ = self.get_embedding(X, i)
        # z_h, _ = self.gru(self.embed_dropout(h_real))
        # z_t, _ = self.gru(self.embed_dropout(t_real))
        # 潜在变量
        z_h = self.eps(self.embed_dropout(h))
        z_t = self.eps(self.embed_dropout(t))
        return h,t,z_h,z_t

    def adversary(self,x,z):
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

    def reparameter(self, mean, std):
        #         sigma = torch.exp(torch.mul(0.5,log_var))
        std_z = torch.randn(std.shape, device=self.device)
        return mean + torch.tensor(self.sigma_parameter).to(self.device) * std * Variable(
            std_z)  # Reparameterization trick

    def transfer_mlp(self, prior, aim='u'):
        # self.transfer_linear_ent = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.transfer_mean_ent = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.transfer_std_ent = nn.Linear(self.embedding_dim, self.embedding_dim)

        transfer_linear = getattr(self, 'transfer_linear_' + aim)
        current_hidden = transfer_linear(prior)
        transfer_mean = getattr(self, 'transfer_mean_' + aim)
        transfer_std = getattr(self, 'transfer_std_' + aim)
        return transfer_mean(current_hidden), transfer_std(current_hidden).mul(0.5).exp()

    def transfer_kl_loss(self, current_mean, current_std, prior_mean, prior_std, dim3=False, aim='u'):
        #  head_trans_loss = self.transfer_kl_loss(head_mean, head_std, head_mean_pri, head_std_pri, False, 'ent')
        dim2 = current_mean.shape[1]
        if (dim3 == False):
            current_transfer_mean = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num ** 2)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std)
        else:
            current_transfer_mean = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std, True)

        return kl_loss

    '''
    KL 误差
    KL(Q(Zt)||P(Zt|B1:t-1))
    P(Zt|B1:t-1) 使用采样计算～～1/K sum_{i=1}^K(P(Zt|Z_{i}t-1))
    '''

    def DKL(self, mean1, std1, mean2, std2, neg=False):
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (mean2 - mean1) * (torch.tensor(1.0, device=self.device) / var2) * (mean2 - mean1)
        tr_std_mul = (torch.tensor(1.0, device=self.device) / var2) * var1
        if (neg == False):
            dkl = (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2).mul(0.5).sum(dim=1).mean()
        else:
            dkl = (torch.log(var2 / var1) - 1 + tr_std_mul + mean_pow2).mul(0.5).sum(dim=2).sum(dim=1).mean()
        return dkl
    def normalize_embeddings(self):
        self.emb_E_real.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E_img.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def log_rank_loss(self, y_pos, y_neg, temp=0):
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma-y_pos
        y_neg = self.gamma-y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        # if self.gpu:
        #     loss = loss.cuda()
        return loss


    def rank_loss(self, y_pos, y_neg):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        if self.gpu:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda()
        else:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss



    def rank_left(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    Xe_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2] if triple[3]>=0 else triple[2]+self.kg.n_relation
                        X_i[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                        Xe_i[i, 0] = i
                        Xe_i[i, 1] = triple[1]
                        Xe_i[i, 2] = triple[2]+self.kg.n_relation if triple[4]>=0 else triple[2]
                        Xe_i[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                    i_score = self.forward(X_i,2) + self.forward(Xe_i,2)
                    if rev_set>0:
                        X_rev = np.ones([self.kg.n_entity,4])
                        Xe_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2 if triple[3]>=0 else triple[2]+self.kg.n_relation+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                            Xe_rev[i, 0] = triple[1]
                            Xe_rev[i, 1] = i
                            Xe_rev[i, 2] = triple[2]+self.kg.n_relation//2+self.kg.n_relation if triple[4]>=0 else triple[2]+self.kg.n_relation//2
                            Xe_rev[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                        i_score = i_score + self.forward(X_rev,2).view(-1) + self.forward(Xe_rev,2).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3],fact[4])]
                    target = i_score[int(triple[0])].clone() * self.thr
                    i_score[filter_out]=1e6 
                    rank_triple=torch.sum((i_score < target ).float()).cpu().item()+1
                    rank.append(rank_triple)
                        

            else:
                ent_ids=list(range(0,self.kg.n_entity))
                num_data=len(X)
                batch_size=30
                num_batch=(num_data+batch_size-1)//batch_size
                num_element=len(X[0])
                for bt in range(num_batch):
                    s=bt*batch_size
                    e=(bt+1)*batch_size
                    batch_X=X[s:e]
                    batch_facts=facts[s:e]

                    cand=ent_ids*len(batch_X)
                    batch_X_i=np.repeat(batch_X,repeats=self.kg.n_entity,axis=0).reshape([-1,num_element])
                    batch_X_i[:,0]=cand
                    batch_i_score=self.forward(batch_X_i,2)
                    if rev_set>0:
                        batch_X_rev=np.repeat(batch_X,repeats=self.kg.n_entity,axis=0).reshape([-1,num_element])
                        batch_X_rev[:,0]=batch_X_i[:,1]
                        batch_X_rev[:,1]=cand
                        batch_X_rev[:, 2] = batch_X_rev[:,2] + self.kg.n_relation // 2
                        batch_i_score=batch_i_score+self.forward(batch_X_rev,2).view(-1)
                    if self.gpu:
                        batch_i_score=batch_i_score.cuda()

                    batch_i_score=batch_i_score.view(-1,self.kg.n_entity)
                    for triple,i_score,fact in zip(batch_X,batch_i_score,batch_facts):
                        filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3],fact[4])]
                        target = i_score[int(triple[0])].clone() * self.thr
                        i_score[filter_out] = 1e6
                        rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                        rank.append(rank_triple)

        return rank

    def rank_right(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    Xe_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2] if triple[3]>=0 else triple[2]+self.kg.n_relation
                        X_i[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                        Xe_i[i, 0] = triple[0] 
                        Xe_i[i, 1] = i
                        Xe_i[i, 2] = triple[2]+self.kg.n_relation if triple[4]>=0 else triple[2]
                        Xe_i[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                    i_score = self.forward(X_i,2) + self.forward(Xe_i,2)
                    if rev_set>0: 
                        X_rev = np.ones([self.kg.n_entity,4])
                        Xe_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2 if triple[3]>=0 else triple[2]+self.kg.n_relation+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                            Xe_rev[i, 0] = i
                            Xe_rev[i, 1] = triple[0]
                            Xe_rev[i, 2] = triple[2]+self.kg.n_relation//2+self.kg.n_relation if triple[4]>=0 else triple[2]+self.kg.n_relation//2
                            Xe_rev[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                        i_score = i_score + self.forward(X_rev,2).view(-1) + self.forward(Xe_rev,2).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3],fact[4])]

                    target = i_score[int(triple[1])].clone() * self.thr
                    i_score[filter_out]=1e6
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
        
                    rank.append(rank_triple)
                    
            else:
                ent_ids=list(range(0,self.kg.n_entity))
                num_data=len(X)
                batch_size=30
                num_batch=(num_data+batch_size-1)//batch_size
                num_element=len(X[0])
                for bt in range(num_batch):
                    s=bt*batch_size
                    e=(bt+1)*batch_size
                    batch_X=X[s:e]
                    batch_facts=facts[s:e]

                    cand=ent_ids*len(batch_X)
                    batch_X_i=np.repeat(batch_X,repeats=self.kg.n_entity,axis=0).reshape([-1,num_element])
                    batch_X_i[:,1]=cand
                    batch_i_score=self.forward(batch_X_i,2)
                    if rev_set>0:
                        batch_X_rev=np.repeat(batch_X,repeats=self.kg.n_entity,axis=0).reshape([-1,num_element])
                        batch_X_rev[:,0]=cand
                        batch_X_rev[:,1]=batch_X_i[:,0]
                        batch_X_rev[:, 2] = batch_X_rev[:,2] + self.kg.n_relation // 2
                        batch_i_score=batch_i_score+self.forward(batch_X_rev,2).view(-1)
                    if self.gpu:
                        batch_i_score=batch_i_score.cuda()

                    batch_i_score=batch_i_score.view(-1,self.kg.n_entity)
                    for triple,i_score,fact in zip(batch_X,batch_i_score,batch_facts):
                        filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3],fact[4])]
                        target = i_score[int(triple[1])].clone() * self.thr
                        i_score[filter_out] = 1e6
                        rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                        rank.append(rank_triple)

        return rank

    def timepred(self, X):
        rank = []
        with torch.no_grad():
            for triple in X:
                X_i = np.ones([self.kg.n_day, len(triple)])
                for i in range(self.kg.n_day):
                    X_i[i, 0] = triple[0]
                    X_i[i, 1] = triple[1]
                    X_i[i, 2] = triple[2]
                    X_i[i, 3:] = self.kg.time_dict[i]
                i_score = self.forward(X_i,2)
                if self.gpu:
                    i_score = i_score.cuda()
    
                target = i_score[triple[3]]           
                rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                rank.append(rank_triple)

        return rank



class AddEps(nn.Module):
    def __init__(self, channels):
        super(AddEps, self).__init__()

        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )

    def forward(self, x):
        eps = torch.randn_like(x)
        eps = self.linear(eps)
        return eps + x

