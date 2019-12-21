#!/bin/env python
# -*- encoding:utf-8 -*-

import os
import sys

import input_data
from data_helper import *
from link_pred_with_emb import linkpred_metrics
from clustering_with_emb import clustering_metrics
from user_item_pred import user_item_pred_metrics

# from sklearn.cluster import KMeans

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
crcan_ui
'''


class CRCAN_UI(nn.Module):
    '''

    '''

    def __init__(self, user_num, item_num, user_attr_num, item_attr_num, hid1_dim, hid2_dim, dropout):
        super(CRCAN_UI, self).__init__()
        self.user_num = user_num

        self.user_attr_num = user_attr_num

        self.item_num = item_num
        self.item_attr_num = item_attr_num

        self.hid1_dim = hid1_dim
        self.hid2_dim = hid2_dim

        self.dropout = dropout

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        # just weight

        self.W_u_1 = nn.Parameter(torch.empty((self.user_num, self.hid1_dim), requires_grad=True))

        self.W_ua_1 = nn.Parameter(torch.empty((self.user_attr_num, self.hid1_dim), requires_grad=True))

        self.W_i_1 = nn.Parameter(torch.empty((self.item_num, self.hid1_dim), requires_grad=True))
        self.W_ia_1 = nn.Parameter(torch.empty((self.item_attr_num, self.hid1_dim), requires_grad=True))

        self.final_user_emb = torch.empty((self.user_num, self.hid1_dim), requires_grad=False)
        self.final_item_emb = torch.empty((self.item_num, self.hid1_dim), requires_grad=False)

    def init_weight(self):

        torch.nn.init.xavier_uniform_(self.W_u_1, gain=1)

        torch.nn.init.xavier_uniform_(self.W_ua_1, gain=1)
        torch.nn.init.xavier_uniform_(self.W_i_1, gain=1)
        torch.nn.init.xavier_uniform_(self.W_ia_1, gain=1)

        torch.nn.init.xavier_uniform_(self.final_user_emb, gain=1)
        torch.nn.init.xavier_uniform_(self.final_item_emb, gain=1)

    def graphconv_sparse(self, A, inputs, W, act):
        x = torch.nn.functional.dropout(inputs, p=self.dropout, training=self.training)
        x = torch.mm(x, W)
        return act(torch.mm(A, x))

    def decode_adj(self, emb):
        latent_adj_nosig = torch.mm(emb, emb.transpose(1, 0))
        return latent_adj_nosig

    def decode_dot(self, emb1, emb2):
        latent_adj_nosig = torch.mm(emb1, emb2.transpose(1, 0))

        return latent_adj_nosig

    def attention_part(self, vec, h, linear=None):
        attn = torch.bmm(vec.detach(), h)
        attention = F.softmax(attn.view(-1, vec.size(1)), dim=1).view(vec.size(0), 1, vec.size(1))
        output_1 = torch.bmm(attention, vec.detach()).view(vec.size(0), vec.size(2))
        if linear is not None:
            output = linear(output_1)
        else:
            output = output_1
        return output, attention

    def attn_ij(self, alpha_w, w_i, w_j):
        concat_tensor = torch.zeros((w_j.shape[0], w_j.shape[1] * 2))
        concat_tensor_1 = torch.zeros((w_j.shape[0], 1))

        for i in range(0, w_j.shape[0]):
            concat_tensor[i] += torch.cat([w_i, w_j[i].view(1, -1)], dim=1)
            concat_tensor_1[i] += torch.mm(concat_tensor[i], alpha_w)

        attention = F.softmax(concat_tensor_1)  # (?, 1)
        output = torch.sum(attention * concat_tensor)
        return output

    def forward(self):
        '''

        '''

        tmp_eps = 1e-16

        # --new attr
        self.user_emb_1 = self.W_u_1
        # self.user_emb_2 = self.W_u_2

        self.user_attr_emb_1 = self.W_ua_1
        self.item_emb_1 = self.W_i_1
        self.item_attr_emb_1 = self.W_ia_1

        # save data
        self.final_user_emb = self.user_emb_1
        self.final_item_emb = self.item_emb_1

        # basic relation
        self.latent_ui_nosig = self.decode_dot(self.user_emb_1, self.item_emb_1)
        self.latent_iu_nosig = self.decode_dot(self.item_emb_1, self.user_emb_1)

        self.latent_uua_nosig = self.decode_dot(self.user_emb_1, self.user_attr_emb_1)
        self.latent_uau_nosig = self.decode_dot(self.user_attr_emb_1, self.user_emb_1)

        self.latent_iia_nosig = self.decode_dot(self.item_emb_1, self.item_attr_emb_1)
        self.latent_iai_nosig = self.decode_dot(self.item_attr_emb_1, self.item_emb_1)

        # composite relation
        self.latent_ui_iia_nosig = torch.mm(torch.mm(self.latent_ui_nosig, self.item_emb_1),
                                            self.item_attr_emb_1.transpose(1, 0))
        self.latent_ui_iu_nosig = torch.mm(torch.mm(self.latent_ui_nosig, self.item_emb_1),
                                           self.user_emb_1.transpose(1, 0))

        self.latent_uua_uau_nosig = torch.mm(torch.mm(self.latent_uua_nosig, self.user_attr_emb_1),
                                             self.user_emb_1.transpose(1, 0))

        self.latent_iu_uua_nosig = torch.mm(torch.mm(self.latent_iu_nosig, self.user_emb_1),
                                            self.user_attr_emb_1.transpose(1, 0))
        self.latent_iu_ui_nosig = torch.mm(torch.mm(self.latent_iu_nosig, self.user_emb_1),
                                           self.item_emb_1.transpose(1, 0))

        self.latent_iia_iai_nosig = torch.mm(torch.mm(self.latent_iia_nosig, self.item_attr_emb_1),
                                             self.item_emb_1.transpose(1, 0))

        self.latent_iai_iia_nosig = torch.mm(torch.mm(self.latent_iai_nosig, self.item_emb_1),
                                             self.item_attr_emb_1.transpose(1, 0))
        self.latent_iai_iu_nosig = torch.mm(torch.mm(self.latent_iai_nosig, self.item_emb_1),
                                            self.user_emb_1.transpose(1, 0))

        self.latent_uau_uua_nosig = torch.mm(torch.mm(self.latent_uau_nosig, self.user_emb_1),
                                             self.user_attr_emb_1.transpose(1, 0))
        self.latent_uau_ui_nosig = torch.mm(torch.mm(self.latent_uau_nosig, self.user_emb_1),
                                            self.item_emb_1.transpose(1, 0))

        return self.latent_ui_nosig, [self.latent_iu_nosig, self.latent_uua_nosig, self.latent_uau_nosig,
                                      self.latent_iia_nosig, self.latent_iai_nosig, self.latent_ui_iia_nosig,
                                      self.latent_ui_iu_nosig, self.latent_uua_uau_nosig, self.latent_iu_uua_nosig,
                                      self.latent_iu_ui_nosig, self.latent_iia_iai_nosig, self.latent_iai_iia_nosig,
                                      self.latent_iai_iu_nosig, self.latent_uau_uua_nosig, self.latent_uau_ui_nosig]


def normalize_sparse_mat(mx):
    """Row-normalize sparse matrix"""
    # rowsum = torch.sum(mx, 0)
    rowsum = torch.sparse.sum(mx, [0])
    r_inv = torch.pow(rowsum, -0.5)
    r_inv = r_inv.to_dense()
    # r_inv[torch.isinf(r_inv)] = 0. #make inf, -inf, nan to 0. #sparse cannot do this operation
    r_inv[torch.isinf(r_inv)] = 0.  # make inf, -inf, nan to 0. #sparse cannot do this operation
    r_mat_inv = torch.diag(r_inv)
    colsum = torch.sparse.sum(mx, [1])
    c_inv = torch.pow(colsum, -0.5)
    c_inv = c_inv.to_dense()

    c_inv[torch.isinf(c_inv)] = 0.
    c_mat_inv = torch.diag(c_inv)

    mx = torch.matmul(mx, r_mat_inv)
    mx = torch.matmul(c_mat_inv, mx)

    return dense_tensor_to_sparse_tensor(mx)


def dense_tensor_to_sparse_tensor(dense):
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1]]  # modify this based on dimensionality
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
        # self.cur_epoch = 0
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
                self.counter = 0
            update_best = False
        else:
            self.best = metric
            self.best_epoch = cur_epoch

            self.counter = 0
            self.extra_info = extra_info
            update_best = True

        # self.cur_epoch += 1

        return dostop, update_best


def read_ui_train_test_pkl(fpath):
    with open(fpath, 'rb') as f:
        (n_users, n_items, item_features, train, valid, test) = pkl.load(
            f)  # here features is not the returned features
    # dok_matrix
    return n_users, n_items, item_features, train, valid, test


def read_mcrec_train_test_pkl(fpath):
    with open(fpath, 'rb') as f:
        trainMatrix, testRatings, testNegatives, user_features, item_features = pkl.load(f)
    # dok_matrix
    return trainMatrix, testRatings, testNegatives, user_features, item_features


def main(opt, cuda, Tensor, LongTensor, ByteTensor, dataset_fea_pkl_fpath):
    # read train data (link prediction task)

    trainMatrix, testRatings, testNegatives, user_features, item_features = read_mcrec_train_test_pkl(
        dataset_fea_pkl_fpath)

    train = trainMatrix
    valid = None
    test = None
    # TODO:is n_items right?
    n_users, n_items = trainMatrix.shape

    train_coo = train.tocoo()
    A_u_i_sp = train_coo  # no norm
    A_u_i = to_torch_sparse_tensor(A_u_i_sp, cuda)

    if item_features is None:
        A_i_ia_sp = scipy.sparse.identity(n_items)
        A_i_ia = to_torch_sparse_tensor(A_i_ia_sp, cuda)
        n_item_attr = n_items
    else:
        A_i_ia_sp = item_features.tocoo()
        A_i_ia = to_torch_sparse_tensor(A_i_ia_sp, cuda)
        n_item_attr = item_features.shape[1]

    if user_features is None:
        A_u_ua_sp = scipy.sparse.identity(n_users)
        A_u_ua = to_torch_sparse_tensor(A_u_ua_sp, cuda)
        n_user_attr = n_users
    else:
        A_u_ua_sp = user_features.tocoo()
        A_u_ua = to_torch_sparse_tensor(A_u_ua_sp, cuda)
        n_user_attr = user_features.shape[1]

    A_u_i_norm = normalize_sparse_mat(A_u_i)
    A_u_ua_norm = normalize_sparse_mat(A_u_ua)
    A_i_ia_norm = normalize_sparse_mat(A_i_ia)

    A_ui_iia = torch.mm(A_u_i_norm.to_dense(), A_i_ia_norm.to_dense())
    A_ui_iu = torch.mm(A_u_i_norm.to_dense(), A_u_i_norm.to_dense().t())
    A_uua_uau = torch.mm(A_u_ua_norm.to_dense(), A_u_ua_norm.to_dense().t())

    A_iu_uua = torch.mm(A_u_i_norm.to_dense().t(), A_u_ua_norm.to_dense())
    A_iu_ui = torch.mm(A_u_i_norm.to_dense().t(), A_u_i_norm.to_dense())
    A_iia_iai = torch.mm(A_i_ia_norm.to_dense(), A_i_ia_norm.to_dense().t())

    A_iai_iia = torch.mm(A_i_ia_norm.to_dense().t(), A_i_ia_norm.to_dense())
    A_iai_iu = torch.mm(A_i_ia_norm.to_dense().t(), A_u_i_norm.to_dense().t())

    A_uau_uua = torch.mm(A_u_ua_norm.to_dense().t(), A_u_ua_norm.to_dense())
    A_uau_ui = torch.mm(A_u_ua_norm.to_dense().t(), A_u_i_norm.to_dense())

    print('user, item, (user_attr), item_attr: %d, %d, %d, %d' % (n_users, n_items, n_user_attr, n_item_attr))

    train_ui_dict = dict()
    test_ui_dict = dict()
    for (edge, value) in train.items():
        u, i = edge
        if u not in train_ui_dict:
            train_ui_dict[u] = [i]
        else:
            train_ui_dict[u].append(i)

    earlystop = EarlyStop(patience=10)

    # TODO:how to calc pos_weight in user-item relation
    # train shape
    # basic 6 relation
    pos_weight_ui = float(A_u_i_sp.shape[0] * A_u_i_sp.shape[1] - A_u_i_sp.sum()) / A_u_i_sp.sum()
    pos_weight_ui = np.ones(A_u_i_sp.shape[1]) * pos_weight_ui
    pos_weight_ui = Tensor(torch.from_numpy(pos_weight_ui).float())
    bce_withlogitloss_ui = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ui)  # weighted ce use

    pos_weight_iu = float(A_u_i_sp.shape[0] * A_u_i_sp.shape[1] - A_u_i_sp.sum()) / A_u_i_sp.sum()
    pos_weight_iu = np.ones(A_u_i_sp.shape[0]) * pos_weight_iu
    pos_weight_iu = Tensor(torch.from_numpy(pos_weight_iu).float())
    bce_withlogitloss_iu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iu)

    pos_weight_uua = float(A_u_ua_sp.shape[0] * A_u_ua_sp.shape[1] - A_u_ua_sp.sum()) / A_u_ua_sp.sum()
    pos_weight_uua = np.ones(A_u_ua_sp.shape[1]) * pos_weight_uua
    pos_weight_uua = Tensor(torch.from_numpy(pos_weight_uua).float())
    bce_withlogitloss_uua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uua)

    pos_weight_iia = float(A_i_ia_sp.shape[0] * A_i_ia_sp.shape[1] - A_i_ia_sp.sum()) / A_i_ia_sp.sum()
    pos_weight_iia = np.ones(A_i_ia_sp.shape[1]) * pos_weight_iia
    pos_weight_iia = Tensor(torch.from_numpy(pos_weight_iia).float())
    bce_withlogitloss_iia = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iia)

    # transpose shape
    pos_weight_iai = float(A_i_ia_sp.shape[0] * A_i_ia_sp.shape[1] - A_i_ia_sp.sum()) / A_i_ia_sp.sum()
    pos_weight_iai = np.ones(A_i_ia_sp.shape[0]) * pos_weight_iai
    pos_weight_iai = Tensor(torch.from_numpy(pos_weight_iai).float())
    bce_withlogitloss_iai = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iai)

    pos_weight_uau = float(A_u_ua_sp.shape[0] * A_u_ua_sp.shape[1] - A_u_ua_sp.sum()) / A_u_ua_sp.sum()
    pos_weight_uau = np.ones(A_u_ua_sp.shape[0]) * pos_weight_uau
    pos_weight_uau = Tensor(torch.from_numpy(pos_weight_uau).float())
    bce_withlogitloss_uau = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uau)

    # composite 10 relations
    pos_weight_ui_iia = float(A_ui_iia.shape[0] * A_ui_iia.shape[1] - float(A_ui_iia.sum())) / float(A_ui_iia.sum())
    pos_weight_ui_iia = np.ones(A_ui_iia.shape[1]) * pos_weight_ui_iia  # pred dimension * value
    pos_weight_ui_iia = Tensor(torch.from_numpy(pos_weight_ui_iia).float())
    bce_withlogitloss_ui_iia = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ui_iia)  # weighted ce use

    pos_weight_ui_iu = float(A_ui_iu.shape[0] * A_ui_iu.shape[1] - float(A_ui_iu.sum())) / float(A_ui_iu.sum())
    pos_weight_ui_iu = np.ones(A_ui_iu.shape[1]) * pos_weight_ui_iu
    pos_weight_ui_iu = Tensor(torch.from_numpy(pos_weight_ui_iu).float())
    bce_withlogitloss_ui_iu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ui_iu)

    pos_weight_uua_uau = float(A_uua_uau.shape[0] * A_uua_uau.shape[1] - float(A_uua_uau.sum())) / float(
        A_uua_uau.sum())
    pos_weight_uua_uau = np.ones(A_uua_uau.shape[1]) * pos_weight_uua_uau
    pos_weight_uua_uau = Tensor(torch.from_numpy(pos_weight_uua_uau).float())
    bce_withlogitloss_uua_uau = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uua_uau)

    pos_weight_iu_uua = float(A_iu_uua.shape[0] * A_iu_uua.shape[1] - float(A_iu_uua.sum())) / float(A_iu_uua.sum())
    pos_weight_iu_uua = np.ones(A_iu_uua.shape[1]) * pos_weight_iu_uua
    pos_weight_iu_uua = Tensor(torch.from_numpy(pos_weight_iu_uua).float())
    bce_withlogitloss_iu_uua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iu_uua)

    pos_weight_iu_ui = float(A_iu_ui.shape[0] * A_iu_ui.shape[1] - float(A_iu_ui.sum())) / float(A_iu_ui.sum())
    pos_weight_iu_ui = np.ones(A_iu_ui.shape[1]) * pos_weight_iu_ui
    pos_weight_iu_ui = Tensor(torch.from_numpy(pos_weight_iu_ui).float())
    bce_withlogitloss_iu_ui = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iu_ui)

    pos_weight_iia_iai = float(A_iia_iai.shape[0] * A_iia_iai.shape[1] - float(A_iia_iai.sum())) / float(
        A_iia_iai.sum())
    pos_weight_iia_iai = np.ones(A_iia_iai.shape[1]) * pos_weight_iia_iai
    pos_weight_iia_iai = Tensor(torch.from_numpy(pos_weight_iia_iai).float())
    bce_withlogitloss_iia_iai = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iia_iai)

    pos_weight_iai_iia = float(A_iai_iia.shape[0] * A_iai_iia.shape[1] - float(A_iai_iia.sum())) / float(
        A_iai_iia.sum())
    pos_weight_iai_iia = np.ones(A_iai_iia.shape[1]) * pos_weight_iai_iia
    pos_weight_iai_iia = Tensor(torch.from_numpy(pos_weight_iai_iia).float())
    bce_withlogitloss_iai_iia = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iai_iia)

    pos_weight_iai_iu = float(A_iai_iu.shape[0] * A_iai_iu.shape[1] - float(A_iai_iu.sum())) / float(A_iai_iu.sum())
    pos_weight_iai_iu = np.ones(A_iai_iu.shape[1]) * pos_weight_iai_iu
    pos_weight_iai_iu = Tensor(torch.from_numpy(pos_weight_iai_iu).float())
    bce_withlogitloss_iai_iu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_iai_iu)

    pos_weight_uau_uua = float(A_uau_uua.shape[0] * A_uau_uua.shape[1] - float(A_uau_uua.sum())) / float(
        A_uau_uua.sum())
    pos_weight_uau_uua = np.ones(A_uau_uua.shape[1]) * pos_weight_uau_uua
    pos_weight_uau_uua = Tensor(torch.from_numpy(pos_weight_uau_uua).float())
    bce_withlogitloss_uau_uua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uau_uua)

    pos_weight_uau_ui = float(A_uau_ui.shape[0] * A_uau_ui.shape[1] - float(A_uau_ui.sum())) / float(A_uau_ui.sum())
    pos_weight_uau_ui = np.ones(A_uau_ui.shape[1]) * pos_weight_uau_ui
    pos_weight_uau_ui = Tensor(torch.from_numpy(pos_weight_uau_ui).float())
    bce_withlogitloss_uau_ui = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uau_ui)

    crcan_ui = CRCAN_UI(n_users, n_items, n_user_attr, n_item_attr, opt.hid1_dim, opt.hid2_dim, opt.dropout)

    crcan_ui.init_weight()

    # --- cuda setting
    if cuda:
        crcan_ui.cuda()

        bce_withlogitloss_ui = bce_withlogitloss_ui.cuda()
        bce_withlogitloss_iu = bce_withlogitloss_iu.cuda()
        bce_withlogitloss_uua = bce_withlogitloss_uua.cuda()
        bce_withlogitloss_uau = bce_withlogitloss_uau.cuda()
        bce_withlogitloss_iia = bce_withlogitloss_iia.cuda()
        bce_withlogitloss_iai = bce_withlogitloss_iai.cuda()

        bce_withlogitloss_ui_iia = bce_withlogitloss_ui_iia.cuda()
        bce_withlogitloss_ui_iu = bce_withlogitloss_ui_iu.cuda()
        bce_withlogitloss_uua_uau = bce_withlogitloss_uua_uau.cuda()
        bce_withlogitloss_iu_uua = bce_withlogitloss_iu_uua.cuda()
        bce_withlogitloss_iu_ui = bce_withlogitloss_iu_ui.cuda()
        bce_withlogitloss_iia_iai = bce_withlogitloss_iia_iai.cuda()
        bce_withlogitloss_iai_iia = bce_withlogitloss_iai_iia.cuda()
        bce_withlogitloss_iai_iu = bce_withlogitloss_iai_iu.cuda()
        bce_withlogitloss_uau_uua = bce_withlogitloss_uau_uua.cuda()
        bce_withlogitloss_uau_ui = bce_withlogitloss_uau_ui.cuda()

    print(crcan_ui)

    # Optimizers: limit parameters to update; be careful, the lr is different
    optimizer_recon_all = torch.optim.Adam(crcan_ui.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    emb_np_for_save = None
    all_batch_index = 0

    def sess_run(epoch):
        '''
        for train 
        '''

        latent_uu_nosig, latent_nosig_list = crcan_ui()

        return latent_uu_nosig, latent_nosig_list

    def epoch_run(opt, epoch, mode='infer'):
        '''
        for infer 
        '''
        assert (mode == 'train' or mode == 'infer')

        if mode == 'train':

            update_save = True
            crcan_ui.train()
        elif mode == 'infer':

            update_save = False
            crcan_ui.eval()

        latent_ui_nosig, latent_nosig_list = sess_run(epoch)

        if mode == 'train':

            latent_iu_nosig, latent_uua_nosig, latent_uau_nosig, latent_iia_nosig, latent_iai_nosig, latent_ui_iia_nosig, latent_ui_iu_nosig, latent_uua_uau_nosig, latent_iu_uua_nosig, latent_iu_ui_nosig, latent_iia_iai_nosig, latent_iai_iia_nosig, latent_iai_iu_nosig, latent_uau_uua_nosig, latent_uau_ui_nosig = latent_nosig_list

            loss_recon_ui = bce_withlogitloss_ui(latent_ui_nosig,
                                                 A_u_i.to_dense())  # (item_batch, user): be careful, use A_u_i not A_u_ib_norm

            loss_recon_iu = bce_withlogitloss_iu(latent_iu_nosig, A_u_i.t().to_dense())

            loss_recon_uua = bce_withlogitloss_uua(latent_uua_nosig, A_u_ua.to_dense())
            loss_recon_uau = bce_withlogitloss_uau(latent_uau_nosig, A_u_ua.t().to_dense())
            loss_recon_iia = bce_withlogitloss_iia(latent_iia_nosig, A_i_ia.to_dense())
            loss_recon_iai = bce_withlogitloss_iai(latent_iai_nosig, A_i_ia.t().to_dense())

            loss_recon_ui_iia = bce_withlogitloss_ui_iia(latent_ui_iia_nosig, A_ui_iia)
            loss_recon_ui_iu = bce_withlogitloss_ui_iu(latent_ui_iu_nosig, A_ui_iu)
            loss_recon_uua_uau = bce_withlogitloss_uua_uau(latent_uua_uau_nosig, A_uua_uau)
            loss_recon_iu_uua = bce_withlogitloss_iu_uua(latent_iu_uua_nosig, A_iu_uua)
            loss_recon_iu_ui = bce_withlogitloss_iu_ui(latent_iu_ui_nosig, A_iu_ui)
            loss_recon_iia_iai = bce_withlogitloss_iia_iai(latent_iia_iai_nosig, A_iia_iai)
            loss_recon_iai_iia = bce_withlogitloss_iai_iia(latent_iai_iia_nosig, A_iai_iia)
            loss_recon_iai_iu = bce_withlogitloss_iai_iu(latent_iai_iu_nosig, A_iai_iu)
            loss_recon_uau_uua = bce_withlogitloss_uau_uua(latent_uau_uua_nosig, A_uau_uua)
            loss_recon_uau_ui = bce_withlogitloss_uau_ui(latent_uau_ui_nosig, A_uau_ui)

            loss_recon_all = loss_recon_ui + loss_recon_iu + loss_recon_uua + loss_recon_uau + loss_recon_iia + loss_recon_iai + loss_recon_ui_iia + loss_recon_ui_iu + loss_recon_uua_uau + loss_recon_iu_uua + loss_recon_iu_ui + loss_recon_iia_iai + loss_recon_iai_iia + loss_recon_iai_iu + loss_recon_uau_uua + loss_recon_uau_ui

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
            user_emb_np = crcan_ui.final_user_emb.data.cpu().numpy()
            item_emb_np = crcan_ui.final_item_emb.data.cpu().numpy()

            return user_emb_np, item_emb_np

    gc.collect()

    print('save network to dir:%s' % opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    test_p_list, test_r_list, test_ndcg_list = [], [], []
    for epoch in range(opt.n_epochs):

        print('%s:enter epoch' % datetime.datetime.now().isoformat())
        sys.stdout.flush()

        epoch_run(opt, epoch, mode='train')
        # get emb_np
        # save result

        user_emb_np_for_save, item_emb_np_for_save = epoch_run(opt, epoch, mode='infer')

        ui_pred_test = user_item_pred_metrics(user_emb_np_for_save, item_emb_np_for_save, train_ui_dict, test_ui_dict,
                                              testRatings, testNegatives)
        test_ps, test_rs, test_ndcgs = ui_pred_test.eval_mcrec_metric(opt.recall_k)
        test_p, test_r, test_ndcg = np.array(test_ps).mean(), np.array(test_rs).mean(), np.array(test_ndcgs).mean()
        test_p_list.append(test_p)
        test_r_list.append(test_r)
        test_ndcg_list.append(test_ndcg)

        print('%s:[Epoch %d/%d] test p:%.4f r:%.4f ndcg:%.4f (@%d)' % (
            datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, test_p, test_r, test_ndcg, opt.recall_k))

        extra_info = 'test p:%f r:%.4f ndcg:%.4f (@%d)' % (test_p, test_r, test_ndcg, opt.recall_k)

        update_best = False
        dostop = False

        if epoch > 0 and test_p > max(test_p_list[:-1]):
            fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch + 1)
            print('update best p at epoch %d, save pkl to:%s' % (epoch + 1, fpath_epoch))

            write_ui_emb_to_pkl_file(user_emb_np_for_save, item_emb_np_for_save, fpath_epoch)

        sys.stdout.flush()

    print('best_test_result: p:%.4f r:%.4f ndcg:%.4f (@%d)' % (
        max(test_p_list), max(test_r_list), max(test_ndcg_list), opt.recall_k))


if __name__ == '__main__':

    # ----command parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--methodname', type=str, required=True, help='methodname ')  #
    parser.add_argument('--save_dir', type=str, required=True, help='save model parameters dir: eg. ../output/vxx/ ')  #
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name: e.g. cora ')  #

    parser.add_argument('--net_type', type=str, required=True, help='net_type: ui')  #

    parser.add_argument('--dataset_fea_pkl', type=str, required=True,
                        help='dataset_fea_pkl: e.g.  ../data/useritem/citeulike/input.pkl')  #

    parser.add_argument('--out_emb_fpath', type=str, required=True,
                        help='out emb fpath: e.g. ../output/vxx/vxxxx.emb ')  #

    # network parameters

    parser.add_argument('--hid1_dim', type=int, required=True, default=100, help='dimensionality of hidden 1 dimension')
    parser.add_argument('--hid2_dim', type=int, required=True, default=100, help='dimensionality of hidden 2 dimension')

    parser.add_argument('--dropout', type=float, required=True, default=0.0,
                        help='dropout value')

    parser.add_argument('--lr', type=float, required=True, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, required=True, default=0.5,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, required=True, default=0.999,
                        help='adam: decay of first order momentum of gradient')

    # learn para
    parser.add_argument('--n_epochs', type=int, required=True, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, required=True, default=1, help='size of the batches')  # origin is 64

    parser.add_argument('--recall_k', type=int, required=True, help='recall@k number: e.g. 50 ')  #
    parser.add_argument('--do_nclu', type=int, required=True, help='do node clustering: e.g. 1:do, 0:do not ')  #

    opt = parser.parse_args()  # option

    dataset_fea_pkl_fpath = opt.dataset_fea_pkl  # '../data/pkl/%s_fea.pkl' % (opt.dataset_name)
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
