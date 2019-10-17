import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from torch.autograd import Variable
from torch.autograd.function import Function

class BCE_sigmoid(nn.Module):
    def __init__(self, size_average = False):
        super(BCE_sigmoid, self).__init__()
        self.size_average = size_average;

    def forward(self, x, labels):
        N = x.size(0);
        mask1 = labels.eq(0);

        mask = 1 - mask1.float();

        target = labels.gt(0);
        target = target.float();

        loss = F.binary_cross_entropy_with_logits(x, target, mask, size_average = False);

        if self.size_average:
            loss = loss/N;

        return loss

class BCE_sigmoid_negtive_bias_all(nn.Module):
    def __init__(self, size_average = False, AU_num = 12, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], database = 0):
        super(BCE_sigmoid_negtive_bias_all, self).__init__()
        self.size_average = size_average;
        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.boundary = 1;

        ##### balance weights for different databases
        if database == 0:
            self.weight = [0.2, 0.3, 0.2, 0.2, 0.5, 0.2, 0.5, 0.2, 0.1, 0.5, 0.2, 0.3];
        elif database == 1:
            self.weight = [0.3, 0.3, 0.5, 0.3, 0.3, 0.5, 0.3, 0.5, 0.3, 0.3, 0.3, 0.3];

        self.balance_a = [];
        for i in range(0,self.AU_num):
            self.balance_a.append(self.weight[self.AU_idx[i]]);

    def forward(self, x, labels):
        N = x.size(0);

        mask1 = labels.eq(0);
        mask = 1 - mask1.float();

        ## selective learning balance
        ################################################################
        for i in range(0, self.AU_num):
            temp = labels[:,i];
            zero_num = torch.sum(temp.eq(0));
            pos_num = torch.sum(temp.eq(1));
            neg_num = torch.sum(temp.eq(-1));
            zero_num = zero_num.float();
            pos_num = pos_num.float();
            neg_num = neg_num.float();
            half_num = (N - zero_num)*self.balance_a[i];

            if (pos_num.data[0] <  half_num.data[0]):
                idx = torch.nonzero(temp.eq(-1));

                sample_num = int(neg_num.data - math.ceil(half_num.data));

                if sample_num < 1:
                    continue;

                zero_idx = random.sample(idx, sample_num);
                for j in range(0, len(zero_idx)):
                    mask[int(zero_idx[j].data), i] = 0;

                ### postive under-representation
                if pos_num.data[0] != 0:
                    ratio = half_num/pos_num;
                    if ratio.data[0] > self.boundary:
                        ratio = self.boundary;

                    idx = torch.nonzero(temp.eq(1));
                    for j in range(0, len(idx)):
                        mask[int(idx[j].data), i] = ratio;
        ################################################################

        target = labels.gt(0);
        target = target.float();

        loss = F.binary_cross_entropy_with_logits(x, target, mask, size_average = False);

        if self.size_average:
            loss = loss/N;

        return loss
