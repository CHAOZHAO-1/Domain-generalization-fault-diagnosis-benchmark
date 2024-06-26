#author:zhaochao time:2023/2/20



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

def EntropyLoss(input_):
    mask = input_.ge(0.000001)# 逐元素比较
    mask_out = torch.masked_select(input_, mask)# 筛选
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings

momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4



def train(model):
    src_iter = iter(src_loader)


    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        GLEARNING_RATE = Glr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(GLEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet1.parameters()},
            {'params': model.sharedNet2.parameters()},
            {'params': model.cls_fc_1.parameters(), 'lr': GLEARNING_RATE},
            {'params': model.cls_fc_2.parameters(), 'lr': GLEARNING_RATE},
            {'params': model.cls_fc_3.parameters(), 'lr': GLEARNING_RATE},
            {'params': model.test_domain_fc.parameters(), 'lr':GLEARNING_RATE},
        ], lr=GLEARNING_RATE / 10, weight_decay=l2_decay)



        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()




        src_pred1,src_pred2,src_pred3,feature,test_domain_pre= model(src_data)


        loss = nn.CrossEntropyLoss()


        test_domain_loss=loss(test_domain_pre, src_label[:,1])



        domian_index_1=(src_label[:,1]==0)
        domian_index_2=(src_label[:,1]==1)
        domian_index_3=(src_label[:,1]==2)

        domain_pred_1=src_pred1[domian_index_1]
        domain_pred_2=src_pred2[domian_index_2]
        domain_pred_3=src_pred3[domian_index_3]

        src_label_1=src_label[domian_index_1][:,0]
        src_label_2=src_label[domian_index_2][:,0]
        src_label_3=src_label[domian_index_3][:,0]

        cls_loss=0

        if len(src_label_1)!=0:
            cls_loss += F.nll_loss(F.log_softmax(domain_pred_1, dim=1), src_label_1)
        if len(src_label_2) != 0:
            cls_loss += F.nll_loss(F.log_softmax(domain_pred_2, dim=1), src_label_2)
        if len(src_label_3) != 0:
            cls_loss += F.nll_loss(F.log_softmax(domain_pred_3, dim=1), src_label_3)

        feature1 = feature[domian_index_1]
        feature2 = feature[domian_index_2]
        feature3 = feature[domian_index_3]

        s1, d = feature1.shape
        s2, d = feature2.shape
        s3, d = feature3.shape

        minlen = min([s1, s2, s3])

        feature1 = feature1[:minlen, :]
        feature2 = feature2[:minlen, :]
        feature3 = feature3[:minlen, :]



        MMD_loss = CORAL(feature1, feature2) \
                   + CORAL(feature1, feature3) \
                   + CORAL(feature2, feature3)

        selector = BatchHardTripletSelector()
        anchor, pos, neg = selector(feature, src_label[:,0])
        triplet_loss = TripletLoss(margin=mar).cuda()
        triplet = triplet_loss(anchor, pos, neg)


        loss = cls_loss + test_domain_loss+MMD_loss+triplet


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tMMD_Loss: {:.6f}\tTri_Loss: {:.6f}\tTEST_domain_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),MMD_loss,triplet,test_domain_loss))

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

            tgt_pred1,tgt_pred2,tgt_pred3,_,_= model(tgt_test_data)


            domian_index_1 = (tgt_test_label[:, 1] == 0)
            domian_index_2 = (tgt_test_label[:, 1] == 1)
            domian_index_3 = (tgt_test_label[:, 1] == 2)

            domain_pred_1 = tgt_pred1[domian_index_1]
            domain_pred_2 = tgt_pred2[domian_index_2]
            domain_pred_3 = tgt_pred3[domian_index_3]

            src_label_1 = tgt_test_label[domian_index_1][:, 0]
            src_label_2 = tgt_test_label[domian_index_2][:, 0]
            src_label_3 = tgt_test_label[domian_index_3][:, 0]

            if len(src_label_1)!=0:
                pred1 = domain_pred_1.data.max(1)[1]
                correct += pred1.eq(src_label_1.data.view_as(pred1)).cpu().sum()

            if len(src_label_2)!=0:
                pred2 = domain_pred_2.data.max(1)[1]
                correct += pred2.eq(src_label_2.data.view_as(pred2)).cpu().sum()

            if len(src_label_3)!=0:
                pred3 = domain_pred_3.data.max(1)[1]
                correct += pred3.eq(src_label_3.data.view_as(pred3)).cpu().sum()


    print('\nset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss



def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    m = nn.Softmax(dim=1)

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            # print(tgt_test_data)

            tgt_test_label = tgt_test_label[:, 0]


            tgt_pred1,tgt_pred2,tgt_pred3,_,test_domain_pre= model(tgt_test_data)

            tgt_domain_pre = m(test_domain_pre)


            pred = m(tgt_pred1) * tgt_domain_pre[:, 0].reshape(-1, 1) + \
                   m(tgt_pred2) * tgt_domain_pre[:, 1].reshape(-1, 1) + m(tgt_pred3) * tgt_domain_pre[:, 2].reshape(-1,1)

            test_loss +=0
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            pred = tgt_pred1.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            pred = tgt_pred2.data.max(1)[1]  # get the index of the max log-probability
            correct3 += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            pred = tgt_pred3.data.max(1)[1]  # get the index of the max log-probability
            correct4 += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()



    print('\nset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy: {}/{} ({:.2f}%), Accuracy: {}/{} ({:.2f}%), Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss,
                                                                               correct1, len(test_loader.dataset),10000. * correct1 / len(test_loader.dataset),
                                                                               correct2, len(test_loader.dataset),10000. * correct2 / len(test_loader.dataset),
                                                                               correct3, len(test_loader.dataset),10000. * correct3 / len(test_loader.dataset),
                                                                               correct4, len(test_loader.dataset),10000. * correct4 / len(test_loader.dataset)
                                                                               ))
    return correct1,test_loss



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 5000
    batch_size = 256
    Glr = 0.0001
    mar=5
    FFT=False

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

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training(dataset1, dataset2, dataset3, batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing(dataset4, batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.DGNIS(num_classes=class_num)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)
            if cuda:
                model.cuda()
            train(model)

































































