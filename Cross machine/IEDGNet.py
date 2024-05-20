#author:zhaochao time:2023/2/21
#author:zhaochao time:2023/2/18


import torch
import torch.nn.functional as F

from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn
import  time
import numpy as np
import  random
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings


momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4



def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)




    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        GLEARNING_RATE = Glr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        DLEARNING_RATE = Dlr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(GLEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': GLEARNING_RATE},
        ], lr=GLEARNING_RATE / 10, weight_decay=l2_decay)

        optimizer_critic = torch.optim.Adam([
            {'params': model.domain_fc.parameters()}
        ], lr=DLEARNING_RATE, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
            tgt_data, tgt_label = tgt_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()

        src_domain_label=src_label[:,1]
        tgt_domain_label=tgt_label[:,1]

        cls_label=src_label[:,0]


        src_pred,src_domain_pred,feature= model(src_data)
        _,tgt_domain_pred,_ = model(tgt_data)

        selector = BatchHardTripletSelector()
        anchor, pos, neg = selector(feature,  cls_label)
        triplet_loss = TripletLoss(margin=1).cuda()
        triplet = triplet_loss(anchor, pos, neg)

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), cls_label)

        loss = nn.CrossEntropyLoss()
        confusion_loss_total = (loss(src_domain_pred, src_domain_label)+loss(tgt_domain_pred, tgt_domain_label))/2



        loss = cls_loss - confusion_loss_total+triplet*1


        optimizer.zero_grad()
        optimizer_critic.zero_grad()


        loss.backward(retain_graph=True)

        confusion_loss_total.backward(retain_graph=True)

        optimizer.step()
        optimizer_critic.step()

        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tconfu_Loss: {:.6f}\tTrip_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),confusion_loss_total,triplet))

        if i % (log_interval * 10) == 0:


            train_correct, train_loss = test_source(model, src_loader)

            test_correct, test_loss = test_target(model, tgt_test_loader)





def test_source(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label=tgt_test_label[:,0]
            # print(tgt_test_data)
            tgt_pred,_,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\nset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label = tgt_test_label[:, 0]
            # print(tgt_test_data)
            tgt_pred,_,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\nset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss





def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 5000
    batch_size = 256
    Glr = 0.001
    Dlr = 0.0001
    AP=1.05
    SNR=10
    FFT = False

    class_num = 3

    datasetlist = [
        ['M_CWRU', 'M_IMS', 'M_JNU', 'M_HUST'],

        ['M_HUST', 'M_JNU', 'M_IMS', 'M_CWRU'],

        ['M_HUST', 'M_IMS', 'M_CWRU', 'M_JNU', ],

        ['M_JNU', 'M_CWRU', 'M_HUST', 'M_IMS'],

        ['M_CWRU', 'M_IMS', 'M_JNU', 'M_SCP'],

        ['M_SCP', 'M_JNU', 'M_IMS', 'M_CWRU'],

        ['M_SCP', 'M_IMS', 'M_CWRU', 'M_JNU'],

        ['M_JNU', 'M_CWRU', 'M_SCP', 'M_IMS'],

        ['M_CWRU', 'M_HUST', 'M_SCP', 'M_XJTU'],

        ['M_CWRU', 'M_HUST', 'M_XJTU', 'M_SCP'],

        ['M_CWRU', 'M_SCP', 'M_XJTU', 'M_HUST'],

        ['M_HUST', 'M_SCP', 'M_XJTU', 'M_CWRU'],

        ['M_CWRU', 'M_JNU', 'M_SCP', 'M_XJTU'],

        ['M_CWRU', 'M_JNU', 'M_XJTU', 'M_SCP'],

        ['M_CWRU', 'M_SCP', 'M_XJTU', 'M_JNU'],

        ['M_JNU', 'M_SCP', 'M_XJTU', 'M_CWRU'],

    ]

    for taskindex in range(16):

        dataset1 = datasetlist[taskindex][0]
        dataset2 = datasetlist[taskindex][1]
        dataset3 = datasetlist[taskindex][2]
        dataset4 = datasetlist[taskindex][3]

        for repeat in range(5):



                cuda = not no_cuda and torch.cuda.is_available()
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.manual_seed(seed)

                kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}



                src_loader = data_loader_1d.load_source_training(dataset1,dataset2,dataset3,batch_size, kwargs)

                tgt_loader = data_loader_1d.load_target_training(AP, SNR, dataset1,dataset2,dataset3,batch_size, kwargs)

                tgt_test_loader = data_loader_1d.load_testing(dataset4, batch_size, kwargs)

                src_dataset_len = len(src_loader.dataset)

                src_loader_len = len(src_loader)
                model = models.IEDG(num_classes=class_num)
                # get_parameter_number(model) 计算模型训练参数个数
                print(model)
                if cuda:
                    model.cuda()
                train(model)












