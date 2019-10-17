import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import scipy.io as sio;

def gen_A_cov(AU_num, AU_idx, database):
    self_AU_num = AU_num;
    self_AU_idx = AU_idx;

    path = './net/';

    if database == 0:
        path = path + 'relation_EmotioNet.mat';
        self_relation_all = sio.loadmat(path)['relation'];
    elif database == 1:
        path = path + 'relation_BP4D.mat';
        self_relation_all = sio.loadmat(path)['relation'];

    self_AU_relation = torch.zeros(self_AU_num, self_AU_num);
    for i in range(0, AU_num):
        for j in range(0, AU_num):
            self_AU_relation[i, j] = self_relation_all[self_AU_idx[i], self_AU_idx[j]];

    _adj = torch.abs(self_AU_relation);

    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_layer(nn.Module):
    def __init__(self, num_classes = 12, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], in_channel = 513, database = 0, weight_relation = 6):
        super(GCN_layer, self).__init__()

        self.num_classes = num_classes;
        self.AU_idx = AU_idx;
        self.database = database;
        self.weight_relation = weight_relation;

        self.gc1 = GraphConvolution(in_channel, 513)
        self.gc2 = GraphConvolution(513, 513)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A_cov(num_classes, AU_idx, database)

        # self.A = Parameter(torch.from_numpy(_adj).float())
        self.A = _adj.cuda();

    def forward(self, x):

        x = x.t();
        adj = gen_adj(self.A).detach()
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]