import torch
import torch.nn as nn

from net.resnet import resnet34;
from net.resnet_gcn import GCN_layer;

class ResNet_GCN_two_views(nn.Module):
    def __init__(self, AU_num = 12, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], output = 2, fusion_mode = 0, database = 0):
        super(ResNet_GCN_two_views, self).__init__()

        self.net1 = resnet34(num_classes = AU_num, num_output=6);
        self.net2 = resnet34(num_classes = AU_num, num_output=6);

        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.fusion_mode = fusion_mode;
        self.output = output;
        self.scale = 1;

        # Different methods to fuse the features of the two views.
        # For fair comparasion, we choose to not fuse the two features.
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            self.fc = nn.Linear(1024, AU_num);
        elif self.fusion_mode == 2:
            self.fc = nn.Linear(512, AU_num);
        elif self.fusion_mode == 3:
            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Linear(512, AU_num)
            )

        self.Is_begin_weight = True;
        self.begin_weight1 = None;
        self.begin_weight2 = None;
        self.relation = GCN_layer(num_classes = AU_num, AU_idx = self.AU_idx, in_channel = 513, database = database);

    def forward(self, data):

        N = data.size(0);

        output1, feat1 = self.net1(data);
        output2, feat2 = self.net2(data);

        weight1 = self.net1.fc.weight;
        bias1 = self.net1.fc.bias;
        weight2 = self.net2.fc.weight;
        bias2 = self.net2.fc.bias;

        bias1 = bias1.view(self.AU_num, -1);
        weight_norm1 = torch.cat((weight1, bias1), 1);
        bias2 = bias2.view(self.AU_num, -1);
        weight_norm2 = torch.cat((weight2, bias2), 1);

        feat_norm1 = feat1;
        feat_norm2 = feat2;

        if self.Is_begin_weight:
            self.begin_weight1 = weight_norm1;
            self.begin_weight2 = weight_norm2;
            self.Is_begin_weight = False;
        else:
            weight_norm1 = self.relation(self.begin_weight1.t());
            weight_norm1 = weight_norm1.t();
            weight_norm2 = self.relation(self.begin_weight2.t());
            weight_norm2 = weight_norm2.t();

        output1 = torch.mm(feat_norm1, torch.t(weight_norm1[:, 0:512])) + weight_norm1[:, 512];
        output1 = self.scale * output1;
        output2 = torch.mm(feat_norm2, torch.t(weight_norm2[:, 0:512])) + weight_norm2[:, 512];
        output2 = self.scale * output2;

        if self.fusion_mode == 0 or self.fusion_mode == 1:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);
        elif self.fusion_mode == 2:
            temp = (feat1 + feat2) / 2;
            output = self.fc(temp);
        elif self.fusion_mode == 3:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);

        if self.output == 1:
            return weight1, bias1, weight2, bias2, feat1, feat2, output1, output2, output
        else:
            return output1, output2, output