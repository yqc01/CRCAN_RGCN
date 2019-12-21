#!/bin/env python
# -*- encoding:utf-8 -*-

import os
import sys

import input_data
from data_helper import *
from link_pred_with_emb import linkpred_metrics
from clustering_with_emb import clustering_metrics
from user_item_pred import user_item_pred_metrics

from sklearn.cluster import KMeans

# from utils import gumbel_softmax

import pickle as pkl

import numpy as np

import argparse
import math
import sys

import random

import gc
# import torchvision.transforms as transforms
# from torchvision.utils import save_image

from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import scipy
import scipy.sparse

# from torchviz import make_dot, make_dot_from_trace

'''
crcan


'''


class CRCAN(nn.Module):
    '''

    '''

    def __init__(self, user_num, user_attr_num, hid1_dim, hid2_dim, dropout):
        super(CRCAN, self).__init__()
        self.user_num = user_num

        self.user_attr_num = user_attr_num

        self.hid1_dim = hid1_dim
        self.hid2_dim = hid2_dim

        self.dropout = dropout

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        # just weight

        self.W_u_1 = nn.Parameter(torch.empty((self.user_num, self.hid1_dim), requires_grad=True))
        self.W_u_2 = nn.Parameter(torch.empty((self.user_num, self.hid1_dim), requires_grad=True))

        self.W_a_1 = nn.Parameter(torch.empty((self.user_attr_num, self.hid1_dim), requires_grad=True))

        self.final_user_emb = torch.empty((self.user_num, self.hid1_dim), requires_grad=False)

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.W_u_1, gain=1)
        torch.nn.init.xavier_uniform_(self.W_u_2, gain=1)

        torch.nn.init.xavier_uniform_(self.W_a_1, gain=1)

        torch.nn.init.xavier_uniform_(self.final_user_emb, gain=1)

    def decode_adj(self, emb):
        latent_adj_nosig = torch.mm(emb, emb.transpose(1, 0))

        return latent_adj_nosig

    def decode_dot(self, emb1, emb2):
        latent_adj_nosig = torch.mm(emb1, emb2.transpose(1, 0))

        return latent_adj_nosig

    def forward(self):
        '''

        '''

        tmp_eps = 1e-16

        # --new attr
        self.user_emb_1 = self.W_u_1
        self.user_emb_2 = self.W_u_2

        self.user_attr_emb_1 = self.W_a_1

        # save data
        self.final_user_emb = self.user_emb_1

        self.latent_uu_nosig = self.decode_adj(self.user_emb_1)

        self.latent_ua_nosig = self.decode_dot(self.user_emb_1, self.user_attr_emb_1)
        self.latent_au_nosig = self.decode_dot(self.user_attr_emb_1, self.user_emb_1)

        self.latent_uuuu_nosig = torch.mm(torch.mm(self.latent_uu_nosig, self.user_emb_1),
                                          self.user_emb_1.transpose(1, 0))

        self.latent_uuua_nosig = torch.mm(torch.mm(self.latent_uu_nosig, self.user_emb_1),
                                          self.user_attr_emb_1.transpose(1, 0))
        self.latent_uaau_nosig = torch.mm(torch.mm(self.latent_ua_nosig, self.user_attr_emb_1),
                                          self.user_emb_1.transpose(1, 0))

        self.latent_auuu_nosig = torch.mm(torch.mm(self.latent_au_nosig, self.user_emb_1),
                                          self.user_emb_1.transpose(1, 0))

        self.latent_auua_nosig = torch.mm(torch.mm(self.latent_au_nosig, self.user_emb_1),
                                          self.user_attr_emb_1.transpose(1, 0))

        return self.latent_uu_nosig, [self.latent_ua_nosig, self.latent_au_nosig, self.latent_uuuu_nosig,
                                      self.latent_uuua_nosig, self.latent_uaau_nosig, self.latent_auuu_nosig,
                                      self.latent_auua_nosig]


def normalize_sparse_mat(mx):
    """Row-normalize sparse matrix"""
    # rowsum = torch.sum(mx, 0)
    rowsum = torch.sparse.sum(mx, [0])
    r_inv = torch.pow(rowsum, -0.5)
    r_inv = r_inv.to_dense()
    # r_inv[torch.isinf(r_inv)] = 0. #make inf, -inf, nan to 0. # sparse cannot do this operation
    r_inv[torch.isinf(r_inv)] = 0.  # make inf, -inf, nan to 0. # sparse cannot do this operation
    r_mat_inv = torch.diag(r_inv)
    # colsum = torch.sum(mx, 1)
    colsum = torch.sparse.sum(mx, [1])
    c_inv = torch.pow(colsum, -0.5)
    c_inv = c_inv.to_dense()

    c_inv[torch.isinf(c_inv)] = 0.
    c_mat_inv = torch.diag(c_inv)

    mx = torch.matmul(mx, r_mat_inv)
    mx = torch.matmul(c_mat_inv, mx)

    return dense_tensor_to_sparse_tensor(mx)
    # return to_torch_sparse_tensor(mx.data.cpu().numpy(), False)


def dense_tensor_to_sparse_tensor(dense):
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1]]
    return torch.sparse.FloatTensor(indices, values, dense.size())


def to_torch_sparse_tensor(M, iscuda):
    '''
    M is a sparse mat (scipy)
    '''
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    if iscuda:
        start_cuda = torch.cuda.FloatTensor([20, 18])  # https://github.com/pytorch/pytorch/issues/8856

        T = torch.cuda.sparse.FloatTensor(indices, values, shape)
    else:
        T = torch.sparse.FloatTensor(indices, values, shape)

    return T


def weights_init_normal(m):
    classname = m.__class__.__name__
    print('init_normal:', classname)
    if classname.find('W') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        # torch.nn.init.constant_(m.bias.data, 0.0)


def batch_generator(batch_size, node_list, graph_dict, all_fea_dict, node_num, attr_num):
    cur_batch = [[], [], []]
    for node in node_list:
        node_adj_index_list = graph_dict[node]
        node_adj_np = np.zeros(node_num, dtype=np.float32)
        node_adj_np[node_adj_index_list] = 1

        node_attr_index_list = all_fea_dict[node]
        node_attr_np = np.zeros(attr_num, dtype=np.float32)
        node_attr_np[node_attr_index_list] = 1

        if len(cur_batch[0]) == batch_size:
            cur_batch = map(lambda x: np.array(x, dtype=np.int64), cur_batch)
            yield cur_batch
            cur_batch = [[], [], []]
        else:
            cur_batch[0].append(node)
            cur_batch[1].append(node_adj_np)
            cur_batch[2].append(node_attr_np)

    if len(cur_batch[0]) > 0:
        cur_batch = map(lambda x: np.array(x, dtype=np.int64), cur_batch)

        yield cur_batch


class EarlyStop(object):
    def __init__(self, patience):
        super(EarlyStop, self).__init__()
        self.patience = patience
        self.best = None
        self.best_epoch = 0
        self.counter = 0
        self.extra_info = None  # record the best valid's test score

    def step(self, metric, extra_info, cur_epoch):
        dostop = False
        update_best = False
        if self.best_epoch == 0:
            self.best = metric
            self.best_epoch = cur_epoch
            self.extra_info = extra_info
            update_best = True

        if metric < self.best:
            self.counter += 1
            if self.counter >= self.patience:
                dostop = True
                self.counter = 0  # changed here, because do not want to see so many log info
            update_best = False
        else:
            self.best = metric
            self.best_epoch = cur_epoch

            self.counter = 0
            self.extra_info = extra_info
            update_best = True

        return dostop, update_best


def read_ui_train_test_pkl(fpath):
    with open(fpath, 'rb') as f:
        (n_users, n_items, item_features, train, valid, test) = pkl.load(
            f)
    # dok_matrix
    return n_users, n_items, item_features, train, valid, test


def main(opt, cuda, Tensor, LongTensor, ByteTensor, dataset_fea_pkl_fpath):
    # read train data (link prediction task)

    feas = read_train_test_pkl(dataset_fea_pkl_fpath)

    if opt.do_nclu == 1:
        infor = {'cora': 7, 'citeseer': 6, 'pubmed': 3}
        cluster_num = infor[opt.dataset_name]
    else:
        pass

    node_num = feas['num_nodes']
    attr_num = feas['num_features']

    val_edges = feas['val_edges']
    val_edges_false = feas['val_edges_false']

    test_edges = feas['test_edges']
    test_edges_false = feas['test_edges_false']

    train_adj_normal_sparse_np = scipy.sparse.coo_matrix((feas['adj_norm'][1], feas['adj_norm'][0].transpose()),
                                                         shape=(node_num, node_num))

    train_adj_label_sparse_np = scipy.sparse.coo_matrix((feas['adj_label'][1], feas['adj_label'][0].transpose()),
                                                        shape=(node_num, node_num))

    attr_sparse_np = scipy.sparse.coo_matrix((feas['features'][1], feas['features'][0].transpose()),
                                             shape=(node_num, attr_num))

    print('user, user_attr: %d, %d' % (node_num, attr_num))

    # A_i_u, A_a_i: sparse
    A_u_u_sp = train_adj_label_sparse_np
    A_u_a_sp = attr_sparse_np

    A_u_u = to_torch_sparse_tensor(A_u_u_sp, cuda)
    A_u_a = to_torch_sparse_tensor(A_u_a_sp, cuda)

    A_u_u_norm = normalize_sparse_mat(A_u_u)
    A_u_a_norm = normalize_sparse_mat(A_u_a)

    A_uuuu = torch.mm(A_u_u_norm.to_dense(), A_u_u_norm.to_dense())
    A_uuua = torch.mm(A_u_u_norm.to_dense(), A_u_a_norm.to_dense())
    A_uaau = torch.mm(A_u_a_norm.to_dense(), A_u_a_norm.to_dense().t())

    A_auua = torch.mm(A_u_a_norm.to_dense().t(), A_u_a_norm.to_dense())
    A_auuu = A_uuua.t()

    earlystop = EarlyStop(patience=10)

    pos_weight = float(A_u_u_sp.shape[0] * A_u_u_sp.shape[1] - A_u_u_sp.sum()) / A_u_u_sp.sum()

    pos_weight_uu = np.ones(node_num) * pos_weight
    pos_weight_uu = Tensor(torch.from_numpy(pos_weight_uu).float())
    bce_withlogitloss_uu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uu)

    pos_weight_a = float(A_u_a_sp.shape[0] * A_u_a_sp.shape[1] - A_u_a_sp.sum()) / A_u_a_sp.sum()

    pos_weight_ua = np.ones(attr_num) * pos_weight_a
    pos_weight_ua = Tensor(torch.from_numpy(pos_weight_ua).float())
    bce_withlogitloss_ua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ua)

    pos_weight_au = np.ones(node_num) * pos_weight_a
    pos_weight_au = Tensor(torch.from_numpy(pos_weight_au).float())
    bce_withlogitloss_au = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_au)

    pos_weight_uuuu = float(A_uuuu.shape[0] * A_uuuu.shape[1] - float(A_uuuu.sum())) / float(A_uuuu.sum())
    pos_weight_uuuu = np.ones(node_num) * pos_weight_uuuu
    pos_weight_uuuu = Tensor(torch.from_numpy(pos_weight_uuuu).float())
    bce_withlogitloss_uuuu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uuuu)

    pos_weight_uuua = float(A_uuua.shape[0] * A_uuua.shape[1] - float(A_uuua.sum())) / float(A_uuua.sum())
    pos_weight_uuua = np.ones(attr_num) * pos_weight_uuua  # pred dimension * value
    pos_weight_uuua = Tensor(torch.from_numpy(pos_weight_uuua).float())
    bce_withlogitloss_uuua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uuua)  # weighted ce use

    pos_weight_uaau = float(A_uaau.shape[0] * A_uaau.shape[1] - float(A_uaau.sum())) / float(A_uaau.sum())
    pos_weight_uaau = np.ones(node_num) * pos_weight_uaau
    pos_weight_uaau = Tensor(torch.from_numpy(pos_weight_uaau).float())
    bce_withlogitloss_uaau = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uaau)

    pos_weight_auuu = float(A_auuu.shape[0] * A_auuu.shape[1] - float(A_auuu.sum())) / float(A_auuu.sum())
    pos_weight_auuu = np.ones(node_num) * pos_weight_auuu
    pos_weight_auuu = Tensor(torch.from_numpy(pos_weight_auuu).float())
    bce_withlogitloss_auuu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_auuu)

    pos_weight_auua = float(A_auua.shape[0] * A_auua.shape[1] - float(A_auua.sum())) / float(A_auua.sum())
    pos_weight_auua = np.ones(attr_num) * pos_weight_auua
    pos_weight_auua = Tensor(torch.from_numpy(pos_weight_auua).float())
    bce_withlogitloss_auua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_auua)

    crcan = CRCAN(node_num, attr_num, opt.hid1_dim, opt.hid2_dim, opt.dropout)

    crcan.init_weight()

    # --- cuda setting
    if cuda:
        crcan.cuda()

        bce_withlogitloss_uu = bce_withlogitloss_uu.cuda()
        bce_withlogitloss_ua = bce_withlogitloss_ua.cuda()
        bce_withlogitloss_au = bce_withlogitloss_au.cuda()

        bce_withlogitloss_uuuu = bce_withlogitloss_uuuu.cuda()
        bce_withlogitloss_uuua = bce_withlogitloss_uuua.cuda()
        bce_withlogitloss_uaau = bce_withlogitloss_uaau.cuda()
        bce_withlogitloss_auuu = bce_withlogitloss_auuu.cuda()
        bce_withlogitloss_auua = bce_withlogitloss_auua.cuda()

    print(crcan)

    optimizer_recon_all = torch.optim.Adam(crcan.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    emb_np_for_save = None
    all_batch_index = 0

    def sess_run(epoch):
        '''
        for train 
        '''

        latent_uu_nosig, latent_nosig_list = crcan()

        return latent_uu_nosig, latent_nosig_list

        # def epoch_run(opt, mode):

    def epoch_run(opt, epoch, mode='infer'):
        '''
        for infer 
        '''
        assert (mode == 'train' or mode == 'infer')

        if mode == 'train':

            update_save = True
            crcan.train()
        elif mode == 'infer':

            update_save = False
            crcan.eval()

        latent_uu_nosig, latent_nosig_list = sess_run(epoch)

        if mode == 'train':

            loss_recon_user_user = bce_withlogitloss_uu(latent_uu_nosig,
                                                        A_u_u.to_dense())
            latent_ua_nosig, latent_au_nosig, latent_uuuu_nosig, latent_uuua_nosig, latent_uaau_nosig, latent_auuu_nosig, latent_auua_nosig = latent_nosig_list

            loss_recon_user_attr = bce_withlogitloss_ua(latent_ua_nosig, A_u_a.to_dense())

            loss_recon_attr_user = bce_withlogitloss_au(latent_au_nosig, A_u_a.t().to_dense())

            loss_recon_uuuu = bce_withlogitloss_uuuu(latent_uuuu_nosig, A_uuuu)
            loss_recon_uuua = bce_withlogitloss_uuua(latent_uuua_nosig, A_uuua)
            loss_recon_uaau = bce_withlogitloss_uaau(latent_uaau_nosig, A_uaau)
            loss_recon_auuu = bce_withlogitloss_auuu(latent_auuu_nosig, A_auuu)
            loss_recon_auua = bce_withlogitloss_auua(latent_auua_nosig, A_auua)

            loss_recon_all = loss_recon_user_user + loss_recon_user_attr + loss_recon_attr_user + loss_recon_uuuu + loss_recon_uuua + loss_recon_uaau + loss_recon_auuu + loss_recon_auua

            optimizer_recon_all.zero_grad()
            loss_recon_all.backward()
            optimizer_recon_all.step()


        elif mode == 'infer':
            pass
        if mode == 'train':
            gc.collect()

            if 'loss_recon_user_attr' in locals():
                print('%s:[Epoch %d/%d] user_user_loss=%f user_attr_loss=%f attr_user_loss=%f ' % (
                    datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs,
                    loss_recon_user_user.data.cpu().numpy(),
                    loss_recon_user_attr.data.cpu().numpy(), loss_recon_attr_user.data.cpu().numpy()))

            else:
                print('%s:[Epoch %d/%d] loss_all=%f ' % (
                    datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_all.data.cpu().numpy()))

            return
        elif mode == 'infer':
            user_emb_np = crcan.final_user_emb.data.cpu().numpy()

            return user_emb_np

    gc.collect()

    print('save network to dir:%s' % opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    best_nclu_acc = 0.0
    best_nclu_info = ''

    lp_dostop_first_str = ''
    for epoch in range(opt.n_epochs):

        print('%s:enter epoch' % datetime.datetime.now().isoformat())
        sys.stdout.flush()

        epoch_run(opt, epoch, mode='train')

        user_emb_np_for_save = epoch_run(opt, epoch, mode='infer')

        if opt.net_type == 'uu':

            lm_test = linkpred_metrics(feas, test_edges, test_edges_false)
            emb_np_for_save = user_emb_np_for_save

            test_auc, test_ap, _ = lm_test.get_roc_score(emb_np_for_save, mode='normal')
            test_result_str = '%s:[Epoch %d/%d] test score auc=%f,ap=%f' % (
                datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, test_auc, test_ap)
            print(test_result_str)
            if epoch + 1 == opt.n_epochs:
                if lp_dostop_first_str == '':
                    lp_dostop_first_str = test_result_str

            lm_val = linkpred_metrics(feas, val_edges, val_edges_false)
            val_auc, val_ap, _ = lm_val.get_roc_score(emb_np_for_save, mode='normal')
            print('%s:[Epoch %d/%d] val score auc=%f,ap=%f' % (
                datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, val_auc, val_ap))
            extra_info = 'test score: auc=%f,ap=%f' % (test_auc, test_ap)

            update_best = False
            dostop = False
            if epoch + 1 >= 50:
                dostop, update_best = earlystop.step(val_auc, extra_info, epoch + 1)
                if dostop == True:
                    lp_dostop_str = "dostop at epoch %d | val auc=%f, best val auc=%f at epoch %d | best %s" % (
                        epoch + 1, val_auc, earlystop.best, earlystop.best_epoch,
                        earlystop.extra_info)  # best_epoch do not +1
                    print(lp_dostop_str)
                    if lp_dostop_first_str == '':
                        lp_dostop_first_str = lp_dostop_str

            epoch_per = 2
            if update_best == True or epoch == 1 or epoch + 1 == opt.n_epochs or (
                    ((epoch + 1) % epoch_per == 0) and ((epoch + 1) <= opt.n_epochs)):
                if epoch == 1 and update_best == False:
                    print('just see epoch 1 result:')
                elif epoch + 1 == opt.n_epochs and update_best == False:
                    print('just see final epoch %d result:' % (epoch + 1))
                elif update_best == False:
                    pass
                else:
                    print('update_best: epoch %d' % (epoch + 1))
                if (((epoch + 1) % epoch_per == 0) and ((epoch + 1) <= opt.n_epochs)):
                    print('for nclu, record the emb')

                # save result and parameters
                fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch + 1)
                write_emb_to_pkl_file(emb_np_for_save, fpath_epoch)

                if opt.do_nclu == 1:  # get the result
                    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(emb_np_for_save)
                    print("Epoch:", '%04d' % (epoch + 1))
                    predict_labels = kmeans.predict(emb_np_for_save)
                    cm = clustering_metrics(feas, feas['true_labels'], predict_labels)
                    cm.evaluationClusterModelFromLabel()

                    ret_score_list = cm.evaluationClusterModelFromLabel()
                    cur_nclu_acc = ret_score_list[3]
                    cur_nclu_str = ret_score_list[0]
                    if cur_nclu_acc > best_nclu_acc:
                        best_nclu_acc = cur_nclu_acc
                        best_nclu_info = 'Epoch:%d:%s' % ((epoch + 1), cur_nclu_str)

        sys.stdout.flush()

    print('%s:writing to  %s and .attr and .attr_new' % (datetime.datetime.now().isoformat(), opt.out_emb_fpath))
    if not os.path.exists(os.path.dirname(opt.out_emb_fpath)):
        os.mkdir(os.path.dirname(opt.out_emb_fpath))

    epoch = opt.n_epochs

    user_emb_np_for_save = epoch_run(opt, epoch, mode='infer')
    emb_np_for_save = user_emb_np_for_save

    fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch)
    write_emb_to_pkl_file(emb_np_for_save, fpath_epoch)
    if opt.do_nclu == 1:  # get the result
        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(emb_np_for_save)
        print("Epoch:", '%04d' % (epoch + 1))
        predict_labels = kmeans.predict(emb_np_for_save)
        cm = clustering_metrics(feas, feas['true_labels'], predict_labels)
        cm.evaluationClusterModelFromLabel()

        print('best acc information:%s' % best_nclu_info)
    print('over')

    print('lp result:%s' % lp_dostop_first_str)


if __name__ == '__main__':

    # ----command parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--methodname', type=str, required=True, help='methodname')  #
    parser.add_argument('--save_dir', type=str, required=True, help='save model parameters dir: eg. ../output/vxx/ ')  #
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name: e.g. cora ')  #

    parser.add_argument('--net_type', type=str, required=True, help='net_type: e.g. uu ')  #

    parser.add_argument('--dataset_fea_pkl', type=str, required=True,
                        help='dataset_fea_pkl: e.g.  ../data/xxx/input.pkl')  #

    parser.add_argument('--out_emb_fpath', type=str, required=True,
                        help='out emb fpath: e.g. ../output/vxx/vxxxx.emb ')  #

    # network parameters

    parser.add_argument('--hid1_dim', type=int, required=True, default=100, help='dimensionality of hidden 1 dimension')
    parser.add_argument('--hid2_dim', type=int, required=True, default=100, help='dimensionality of hidden 2 dimension')

    parser.add_argument('--dropout', type=float, required=True, default=0.0,
                        help='dropout value')

    # adam para
    parser.add_argument('--lr', type=float, required=True, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, required=True, default=0.5,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, required=True, default=0.999,
                        help='adam: decay of first order momentum of gradient')

    # learn para
    parser.add_argument('--n_epochs', type=int, required=True, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, required=True, default=1, help='size of the batches')

    parser.add_argument('--recall_k', type=int, required=True, help='recall@k number: e.g. 50 ')
    parser.add_argument('--do_nclu', type=int, required=True, help='do node clustering: e.g. 1:do, 0:do not ')  #

    opt = parser.parse_args()  # option

    dataset_fea_pkl_fpath = opt.dataset_fea_pkl  
    print("arguments:%s" % opt)
    print("read feature(uu / ui, attr) pkl path:%s" % dataset_fea_pkl_fpath)

    if opt.do_nclu == 1:
        print('do node_clustering')
    elif opt.do_nclu == 0:
        print('do not do node clustering')
    else:
        print('unknown --do_nclu, exit')
        exit()
    cuda = False

    Tensor = lambda x: torch.cuda.FloatTensor(x.cuda()) if cuda else torch.FloatTensor(x)
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    print("cuda:%s" % cuda)
    print("Tensor:%s" % Tensor)

    if opt.net_type not in ['uu', 'ui']:
        print('unknown net_type')
        exit()
    # set seed
    seednum = 1
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    random.seed(seednum)

    main(opt, cuda, Tensor, LongTensor, ByteTensor, dataset_fea_pkl_fpath)
