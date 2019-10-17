import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as sio

from PIL import Image

sys.path.append('.');

from loss.loss_multi_view_final import MultiView_all_loss;
from net.resnet_multi_view import ResNet_GCN_two_views;

batch_size_num = 10;
epoch_num = 30;
learning_rate = 0.001;
weight_decay = 0;
test_batch_size = 20;
unlabel_num = 50000;

##################################
########## parameters ############
##################################
### fusion_mode: 0 no fusion / 1 concate / 2 mean / 3 concate two layer
### database: 0 EmotioNet / 1 BP4D
### use_web: 0 only test images/ 1 training with unlabeled images
### lambda_co_regularization:
### AU_idx: the AU idxes you want to consider
##################################
fusion_mode = 0;
database = 0;
use_web = 0;

AU_num = 12;
AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11];

lambda_co_regularization = 100;
lambda_multi_view = 400;
###################################

###################################
########## transforms #############
###################################
## transform for training
transform_train = transforms.Compose([
    transforms.Resize(240),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])

## transform for testing
transform_test = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])
#####################################

###################################
########## network ################
###################################
net = ResNet_GCN_two_views(AU_num=AU_num, AU_idx=AU_idx, output=1, fusion_mode=fusion_mode, database=database);
model_path = './model/EmotioNet_model.pth.tar';
temp = torch.load(model_path);
net.load_state_dict(temp['net'])
net.cuda();
####################################

###################################
########## lossfunction ###########
###################################
# loss function combination all losses, including the L_mv, L_cr and the BCE loss for AU classification
# Input: AU_num, AU_idx, fusion_mode, use_web and database are explained in the parameters section

# Explaination of the BCE loss: We choose the modified Selective Learning BCE loss for supervision.
# Since the positive/negative ratio is very small for some AUs, we only consider the positive under-representation situation
# and set a boundary for all AUs.
###################################
lossfunc = MultiView_all_loss(AU_num = AU_num, AU_idx = AU_idx, fusion_mode = fusion_mode,
                              use_web = use_web, database = database,
                              lambda_co_regularization = lambda_co_regularization, lambda_multi_view = lambda_multi_view);
###################################

def transfrom_img_test(img):
    img = transform_test(img);

    return img;

def main():
    img = Image.open('./test_img/test_img.jpg').convert('RGB');
    img = transfrom_img_test(img);

    img = Variable(img).cuda();
    img = img.view(1, img.size(0), img.size(1), img.size(2));

    AU_view1, AU_view2, AU_fusion = net(img);
    AU_view1 = torch.sigmoid(AU_view1);
    AU_view2 = torch.sigmoid(AU_view2);
    print(AU_view1)
    print(AU_view2)

main();
