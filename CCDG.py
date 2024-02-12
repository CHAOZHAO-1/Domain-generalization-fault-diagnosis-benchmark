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


    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        cls_label=src_label[:,0]

        domain_label=src_label[:,1]

        optimizer.zero_grad()
        src_pred,src_feature= model(src_data)

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), cls_label)

        contrasitve_loss = contrastive_loss(src_feature, domain_label, 0.7)

        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1


        loss = cls_loss+contrasitve_loss*lambd
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tcontrastive_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),contrasitve_loss.item()))

        if i % (log_interval * 10) == 0:




            train_correct, train_loss = test_source(model, src_loader)


            test_correct, test_loss = test_target(model, tgt_test_loader)




def contrastive_loss(domains_features, domains_labels, temperature):
    # masking for the corresponding class labels.
    device= "cuda" if torch.cuda.is_available() else "cpu"

    anchor_feature = domains_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    labels = domains_labels
    labels= labels.contiguous().view(-1, 1)
    # Generate masking for positive and negative pairs.
    mask = torch.eq(labels, labels.T).float().to(device)
    # Applying contrastive via product among the features
    # Measure the similarity between all samples in the batch
    # reiterate fact from Linear Algebra if u and v two vectors are normalised implies cosine similarity
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

    # for numerical stability
    # substract max value from the output
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # create inverted identity matrix with same shape as mask.
    # only diagnoal is zeros and all others are ones
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device),
                                0)
    # mask-out self-contrast cases
    # the diagnoal represent same samples similarity i=j>> we need to remove this
    # remove the true labels mask
    # all ones in mask would be only samples with same labels
    mask = mask * logits_mask

    # compute log_prob and remove the diagnal
    # remove same features from the normalized contrastive matrix
    # The denominoator of the equation samples
    exp_logits = torch.exp(logits) * logits_mask

    # substract the whole multiplications from the negative pair and positive pairs
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mask_sum = mask.sum(1)
    zeros_idx = torch.where(mask_sum == 0)[0]
    mask_sum[zeros_idx] = 1

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    # loss
    loss = (- 1 * mean_log_prob_pos)
    loss = loss.mean()

    return loss

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
            tgt_pred,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name,test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
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
            # print(tgt_test_data)
            tgt_pred,_= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name,test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)
    iteration = 5000
    batch_size = 256
    lr = 0.001
    FFT = False


    CWRUTasksetting = {'dataset': 'C-CWRU', 'class_num': 10, 'src_tar': np.array([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2,3,  0]])}

    LWTasksetting = {'dataset': 'C-LW', 'class_num': 4, 'src_tar':np.array([[2, 3, 4, 5], [2, 3, 5, 4], [2, 4, 5, 3], [3, 4, 5,2]])}

    PUTasksetting = {'dataset': 'C-PU', 'class_num': 12,'src_tar': np.array([[6, 7, 8, 9], [6, 7, 9, 8], [6, 8, 9, 7], [7, 8,9,6]])}

    PHMTasksetting = {'dataset': 'C-PHM', 'class_num': 6,'src_tar': np.array([[9, 10, 11, 12],[9, 10, 12, 11],[9, 11,12, 10],[10, 11, 12, 9]])}

    SQbearingTasksetting = {'dataset': 'C-SQbearing', 'class_num': 9,'src_tar': np.array([[6, 7, 8, 9], [6, 7, 9, 8], [6, 8, 9, 7], [7, 8,9,6]])}

    SQgearboxTasksetting = {'dataset': 'C-SQgearbox', 'class_num': 3,'src_tar': np.array([[0, 7, 14, 21], [0, 7, 21, 14], [0, 14, 21,  7], [7, 14, 21, 0 ]])}



    Tasksetting =[CWRUTasksetting,LWTasksetting,
                  PUTasksetting,PHMTasksetting,
                  SQbearingTasksetting,SQgearboxTasksetting
                  ]



    for tasknumber in range(6):

        dataset=Tasksetting[tasknumber]['dataset']

        class_num=Tasksetting[tasknumber]['class_num']

        src_tar=Tasksetting[tasknumber]['src_tar']


        for taskindex in range(4):
            source1 = src_tar[taskindex][0]
            source2 = src_tar[taskindex][1]
            source3 = src_tar[taskindex][2]
            target = src_tar[taskindex][3]
            src = src_tar[taskindex][:-1]

            for repeat in range(10):

                root_path = '/home/zhaochao/research/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'

                src_name1 = 'load' + str(source1) + '_train'
                src_name2 = 'load' + str(source2) + '_train'
                src_name3 = 'load' + str(source3) + '_train'

                tgt_name = 'load' + str(target) + '_train'
                test_name = 'load' + str(target) + '_test'

                cuda = not no_cuda and torch.cuda.is_available()
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.manual_seed(seed)

                kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}

                src_loader = data_loader_1d.load_training(root_path, src_name1, src_name2, src_name3, src, FFT,
                                                          class_num,
                                                          batch_size, kwargs)
                tgt_test_loader = data_loader_1d.load_testing(root_path, test_name, FFT, class_num,
                                                              batch_size, kwargs)

                src_dataset_len = len(src_loader.dataset)

                src_loader_len = len(src_loader)
                model = models.CNN_1D(num_classes=class_num)
                # get_parameter_number(model) 计算模型训练参数个数
                print(model)
                if cuda:
                    model.cuda()
                train(model)
































