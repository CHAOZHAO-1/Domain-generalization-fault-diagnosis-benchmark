
import torch

import torch.nn as nn
from utils import *

import torch.nn.functional as F
import torch.autograd as autograd

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)



class CNN_1D(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1D, self).__init__()
        # self.sharedNet = resnet18(False)
        # self.cls_fc = nn.Linear(512, num_classes)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source):

        # source= source.unsqueeze(1)

        feature = self.sharedNet(source)
        source=self.cls_fc(feature)

        return source,feature




class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )





        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x



class AdversarialNetwork_multi(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork_multi, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_layer2 = nn.Linear(128, 3)
        # self.ad_layer1.weight.data.normal_(0, 0.1)
        # self.ad_layer2.weight.data.normal_(0, 0.3)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # print(x.size())
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.softmax(x)
        return x

    def output_num(self):
        return 1





class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_layer2 = nn.Linear(128, 2)
        # self.ad_layer1.weight.data.normal_(0, 0.1)
        # self.ad_layer2.weight.data.normal_(0, 0.3)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # print(x.size())
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.softmax(x)
        return x

    def output_num(self):
        return 1


class DeepALL_ADV(nn.Module):

    def __init__(self, num_classes=31):
        super(DeepALL_ADV, self).__init__()


        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)
        self.domain_fc = AdversarialNetwork_multi(in_feature=256)


    def forward(self, source):

        feature = self.sharedNet(source)
        label_pre=self.cls_fc(feature)
        domain_pre = self.domain_fc(feature)

        return label_pre, domain_pre


class IEDG(nn.Module):

    def __init__(self, num_classes=31):
        super(IEDG, self).__init__()


        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)
        self.domain_fc = AdversarialNetwork(in_feature=256)


    def forward(self, source):

        feature = self.sharedNet(source)

        label_pre=self.cls_fc(feature)
        domain_pre = self.domain_fc(feature)

        return label_pre, domain_pre,feature




class DGNIS(nn.Module):

    def __init__(self, num_classes=31):
        super(DGNIS, self).__init__()


        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()

        self.cls_fc_1 = nn.Linear(256, num_classes)
        self.cls_fc_2 = nn.Linear(256, num_classes)
        self.cls_fc_3 = nn.Linear(256, num_classes)

        self.test_domain_fc = nn.Linear(256, 3)


    def forward(self, source):


        feature1 = self.sharedNet1(source)
        feature2 = self.sharedNet2(source)

        pre_1=self.cls_fc_1(feature1)
        pre_2 = self.cls_fc_2(feature1)
        pre_3 = self.cls_fc_3(feature1)


        test_domian_pre = self.test_domain_fc(feature2)

        return pre_1,pre_2,pre_3,feature1,test_domian_pre