#author:zhaochao time:2021/5/18

import torch as t
import torch.nn.functional as F
import numpy as np
import  random
import torch.nn as nn
import scipy.io as scio
from scipy.fftpack import fft

import copy

def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = t.mean(source, 1, keepdim=True) - source
    xc = t.matmul(t.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = t.mean(target, 1, keepdim=True) - target
    xct = t.matmul(t.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = t.mean(t.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*4)
    return loss
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = t.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = t.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [t.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = t.mean(XX + YY - XY -YX)

    return loss

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = t.div(w_avg[key], len(w))
    return w_avg

def wgn(x, snr):
    Ps = np.sum(abs(x)**2,axis=1)/len(x)
    Pn = Ps/(10**((snr/10)))
    row,columns=x.shape
    Pn = np.repeat(Pn.reshape(-1,1),columns, axis=1)

    noise = np.random.randn(row,columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def zscore(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Z = (Z - Zmin.reshape(-1,1)) / (Zmax.reshape(-1,1) - Zmin.reshape(-1,1))
    return Z


def min_max(Z):
    Zmin = Z.min(axis=1)

    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = t.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = t.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = t.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = t.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = t.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()# 计算距离
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)#返回对角线索引
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / t.norm(x2, dim=1)
    return sim




def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = t.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd



def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * t.log(predict_prob + epsilon)
    return t.sum(instance_level_weight * ce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    mask = predict_prob.ge(0.000001)  # 逐元素比较
    mask_out = t.masked_select(predict_prob, mask)#


    entropy =-mask_out * t.log(mask_out)


#
    return t.sum(instance_level_weight * entropy * class_level_weight) / float(N)

#
# def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
#     N, C = predict_prob.size()
#
#     if class_level_weight is None:
#         class_level_weight = 1.0
#     else:
#         if len(class_level_weight.size()) == 1:
#             class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
#         assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
#
#     if instance_level_weight is None:
#         instance_level_weight = 1.0
#     else:
#         if len(instance_level_weight.size()) == 1:
#             instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
#         assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'
#
#     entropy = -predict_prob * t.log(predict_prob + epsilon)
#
#     return t.sum(entropy ,dim=1)


def log(name1,name2,Train_Loss_list,Train_Accuracy_list,Test_Loss_list,Test_Accuracy_list,Train_Time):
    f = open('./'+name1+'/'+name2+'.txt', 'w')

    f.write('train_loss:')
    f.write(str(Train_Loss_list))
    f.write('\r\n')
    f.write('train_acc:')
    f.write(str(Train_Accuracy_list))
    f.write('\r\n')
    f.write('test_loss:')
    f.write(str(Test_Loss_list))
    f.write('\r\n')
    f.write('test_acc:')
    f.write(str(Test_Accuracy_list))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time))
    f.close()


def normalization(input):
    kethe=0.0000000001
    output=(input-min(input))/(max(input)-min(input)+kethe)

    return output




def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None,
                                  epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'


    bce = -label * t.log(predict_prob + epsilon) - (1.0 - label) * t.log(1.0 - predict_prob + epsilon)

    return t.sum(instance_level_weight * bce * class_level_weight) / float(N)


class Center_loss(nn.Module):
    def __init__(self,src_class):
        super(Center_loss, self).__init__()

        self.n_class=src_class
        self.MSELoss = nn.MSELoss()  # (x-y)^2
        self.MSELoss = self.MSELoss.cuda()



    def forward(self, s_feature,s_labels):


        n, d = s_feature.shape

        # get labels


        # image number in each class
        ones = t.ones_like(s_labels, dtype=t.float)
        zeros = t.zeros(self.n_class)

        zeros = zeros.cuda()

        s_n_classes = zeros.scatter_add(0, s_labels, ones)


        # image number cannot be 0, when calculating centroids
        ones = t.ones_like(s_n_classes)
        s_n_classes = t.max(s_n_classes, ones)


        # calculating centroids, sum and divide
        zeros = t.zeros(self.n_class, d)

        zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, t.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)

        s_centroid = t.div(s_sum_feature, s_n_classes.view(self.n_class, 1))


        # calculating inter distance

        temp = t.zeros((n, d)).cuda()

        for i in range(n):
            temp[i] = s_centroid[s_labels[i]]

       #
        # intra_loss = t.norm(temp-s_feature, p=1, dim=0).sum()
        # intra_loss = intra_loss / (d * n)

        #### way 1:
        intra_loss = self.MSELoss(temp, s_feature)



        return intra_loss


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')

    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def get_dataset(args,client):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """



    root_path = '/home/zhaochao/research/DTL/data/' + args.dataset + 'data' + str(args.class_num) + '.mat'

    data = scio.loadmat(root_path)

    train_loaders = {}


    for k in range(3):
        if args.fft1 == True:
            train_fea = zscore((min_max(abs(fft(data[client[k]]))[:, 0:512])))
        if args.fft1 == False:
            train_fea = zscore(data[client[k]])
    #
        train_label = t.zeros((800 * args.class_num))
        for i in range(800 * args.class_num):
            train_label[i] = i // 800

        print(train_fea.shape)
        print(train_label.shape)
    # #
        train_label = train_label.long()
        train_fea = t.from_numpy(train_fea)
        train_fea = t.tensor(train_fea, dtype=t.float32)
        data_s = t.utils.data.TensorDataset(train_fea, train_label)
        train_loaders[k] = t.utils.data.DataLoader(data_s, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                       num_workers=args.workers, pin_memory=args.pin)

    #
    if args.fft1 == True:
        test_fea = zscore((min_max(abs(fft(data[client[3]]))[:, 0:512])))
    if args.fft1 == False:
        test_fea = zscore(data[client[3]])

    test_label = t.zeros((200 * args.class_num))
    for i in range(200 * args.class_num):
        test_label[i] = i // 200

    print(test_fea.shape)

    test_label = test_label.long()
    test_fea = t.from_numpy(test_fea)
    test_fea = t.tensor(test_fea, dtype=t.float32)
    data_t = t.utils.data.TensorDataset(test_fea, test_label)
    test_loader = t.utils.data.DataLoader(data_t, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                               num_workers=args.workers, pin_memory=args.pin)

    return train_loaders, test_loader