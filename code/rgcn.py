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

from utils import gumbel_softmax

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
rgcn

'''


class RGCN(nn.Module):
    '''

    '''

    def __init__(self, user_num, user_attr_num, recon_adj_hid1_dim, recon_adj_hid2_dim, recon_adj_hid3_dim,
                 recon_attr_hid1_dim, recon_attr_hid2_dim, recon_attr_hid3_dim, dropout):
        super(RGCN, self).__init__()
        self.user_num = user_num

        self.user_attr_num = user_attr_num

        self.recon_adj_hid1_dim = recon_adj_hid1_dim
        self.recon_adj_hid2_dim = recon_adj_hid2_dim
        self.recon_adj_hid3_dim = recon_adj_hid3_dim

        self.recon_attr_hid1_dim = recon_attr_hid1_dim
        self.recon_attr_hid2_dim = recon_attr_hid2_dim
        self.recon_attr_hid3_dim = recon_attr_hid3_dim

        self.dropout = dropout

        self.uuuu_max = None
        self.uuua_max = None
        self.uaau_max = None

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        # just weight
        self.W_w_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_b = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ula_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ula_b = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ala_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ani_a = nn.Parameter(torch.empty((1), requires_grad=True))

        self.Wauuu = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wuuuu = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wuuua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wauua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wuaau = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))

        self.H_attention_u = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim, 1), requires_grad=True))
        self.H_attention_a = nn.Parameter(
            torch.empty((self.user_attr_num, self.recon_attr_hid2_dim, 1), requires_grad=True))
        self.H_attention_ua_uaa = nn.Parameter(
            torch.empty((self.user_num, self.recon_attr_hid2_dim, 1), requires_grad=True))
        self.H_attention_u_12 = nn.Parameter(
            torch.empty((self.user_num, self.recon_adj_hid2_dim, 1), requires_grad=True))

        self.W_user_aa = nn.Parameter(
            torch.empty((self.recon_attr_hid2_dim, self.recon_attr_hid2_dim), requires_grad=True))
        self.W_anew = nn.Parameter(torch.empty((self.recon_attr_hid2_dim, self.recon_adj_hid2_dim), requires_grad=True))

        self.linear_a_new = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim)

        self.linear_u = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim)
        self.linear_a = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim)
        self.linear_ua_uaa = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim)
        self.linear_u_12 = torch.nn.Linear(self.recon_adj_hid2_dim, self.recon_adj_hid2_dim)

        self.linear_final = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim)
        self.W_ori_uuua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_adj_hid2_dim), requires_grad=True))
        self.W_h_e = nn.Parameter(torch.empty((self.recon_adj_hid2_dim, self.recon_adj_hid2_dim), requires_grad=True))

        self.final_user_emb = torch.empty((self.user_num, self.recon_adj_hid2_dim),
                                          requires_grad=False)  # TODO: may need gradient later?
        self.final_user_attr_emb = torch.empty((self.user_attr_num, self.recon_adj_hid2_dim), requires_grad=False)  #

    def init_weight(self):

        torch.nn.init.xavier_uniform_(self.Wauuu, gain=1)
        torch.nn.init.xavier_uniform_(self.Wuuuu, gain=1)
        torch.nn.init.xavier_uniform_(self.Wuuua, gain=1)
        torch.nn.init.xavier_uniform_(self.Wauua, gain=1)
        torch.nn.init.xavier_uniform_(self.Wuaau, gain=1)

        torch.nn.init.xavier_uniform_(self.H_attention_u, gain=1)
        torch.nn.init.xavier_uniform_(self.H_attention_a, gain=1)

        torch.nn.init.xavier_uniform_(self.H_attention_ua_uaa, gain=1)
        torch.nn.init.xavier_uniform_(self.H_attention_u_12, gain=1)

        torch.nn.init.xavier_uniform_(self.linear_a_new.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_a.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_u.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_ua_uaa.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_u_12.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_final.weight, gain=1)

        torch.nn.init.xavier_uniform_(self.W_anew, gain=1)
        torch.nn.init.xavier_uniform_(self.W_user_aa, gain=1)

        torch.nn.init.xavier_uniform_(self.W_ori_uuua, gain=1)
        torch.nn.init.xavier_uniform_(self.W_h_e, gain=1)

        torch.nn.init.xavier_uniform_(self.final_user_emb, gain=1)
        torch.nn.init.xavier_uniform_(self.final_user_attr_emb, gain=1)

        torch.nn.init.constant_(self.W_w_a, 0.33)
        torch.nn.init.constant_(self.W_w_b, 0.33)
        torch.nn.init.constant_(self.W_w_ula_a, 0.33)
        torch.nn.init.constant_(self.W_w_ula_b, 0.33)
        torch.nn.init.constant_(self.W_w_ala_a, 0.5)
        torch.nn.init.constant_(self.W_w_ani_a, 0.5)

    def graphconv_sparse(self, A, inputs, W, act):
        x = torch.nn.functional.dropout(inputs, p=self.dropout, training=self.training)
        x = torch.mm(x, W)  # (node_num, W.shape[1])
        return act(torch.mm(A, x))

    def graphconv(self, A, inputs, W, act):
        x = torch.nn.functional.dropout(inputs, p=self.dropout, training=self.training)
        x = torch.mm(x, W)  # (node_num, W.shape[1])
        return act(torch.mm(A, x))

    def decode_adj(self, emb):
        latent_adj_nosig = torch.mm(emb, emb.transpose(1, 0))

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

    def forward(self, x_u_a, x_u_u, A_u_a, A_u_u, std, tao, update_save=True):
        '''
        input:

        x_u_a: sparse : 0.x (generate from user-item)
        x_i_a: sparse : 0/1 (generate from actual item-attr)
        x_ib_a:(item_batch, attr) :change to sparse and the whole u-i relation
        A_u_ib:(user, item_batch) :the same: sparse
        A_i_u:(item, user) sparse
        A_a_i:(attr, item) sparse

        return:
        adj_embeddings: (N, hid2)
        latent_adj_nosig: (batch, N)
        latent_attr_nosig: (batch, attr)
        attr_new_input: (batch, attr)

        latent_ui_nosig:(user, item_batch)
        self.latent_attr_nosig:(item_batch, attr)
        attr_new_input: (item_batch, attr)
        '''

        tmp_eps = 1e-16

        self.user_emb_uuuu = self.graphconv_sparse(A_u_u, x_u_u, self.Wuuuu, lambda x: x)

        self.user_emb_uuua = self.graphconv_sparse(A_u_u, x_u_a, self.Wuuua, lambda x: x)

        self.user_emb_uaau = self.graphconv_sparse(A_u_a, x_u_a.t(), self.Wuaau, lambda x: x)
        self.user_emb_uaau = self.user_emb_uaau / (
                    tmp_eps + self.user_emb_uaau.norm(p=2, dim=1, keepdim=True).expand_as(self.user_emb_uaau))

        self.attr_emb_auuu = self.graphconv_sparse(A_u_a.t(), x_u_u, self.Wauuu, lambda x: x)
        self.attr_emb_auuu = self.attr_emb_auuu / (
                    self.attr_emb_auuu.norm(p=2, dim=1, keepdim=True).expand_as(self.attr_emb_auuu) + tmp_eps)
        self.attr_emb_auua = self.graphconv_sparse(A_u_a.t(), x_u_a, self.Wauua, lambda x: x)
        self.attr_emb_auua = self.attr_emb_auua / (
                    self.attr_emb_auua.norm(p=2, dim=1, keepdim=True).expand_as(self.attr_emb_auua) + tmp_eps)

        self.user_latent_attr = self.W_w_ula_a * self.user_emb_uuuu + self.W_w_ula_b * self.user_emb_uuua + (
                    1.0 - self.W_w_ula_a - self.W_w_ula_b) * self.user_emb_uaau

        self.attr_latent_attr = self.W_w_ala_a * self.attr_emb_auuu + (1.0 - self.W_w_ala_a) * self.attr_emb_auua

        self.user_attr_attr = self.graphconv_sparse(A_u_a, self.attr_latent_attr, self.W_user_aa,
                                                    lambda x: x)  # user's attr's attr

        self.user_attr_attr = self.user_attr_attr / (
                    self.user_attr_attr.norm(p=2, dim=1, keepdim=True).expand_as(self.user_attr_attr) + tmp_eps)

        attr_new_input = self.W_w_ani_a * self.user_latent_attr + (1.0 - self.W_w_ani_a) * self.user_attr_attr

        # --new attr
        self.user_emb_new_1 = self.graphconv_sparse(A_u_u, attr_new_input, self.W_anew,
                                                    lambda x: x)  # TODO:activation needs changing: sparse
        self.user_emb_new_1 = self.user_emb_new_1 / (
                    tmp_eps + self.user_emb_new_1.norm(p=2, dim=1, keepdim=True).expand_as(self.user_emb_new_1))

        self.user_emb_new_2 = self.graphconv_sparse(A_u_u, x_u_a, self.W_ori_uuua, lambda x: x)

        self.user_emb_new = self.W_w_a * self.user_emb_new_1 + self.W_w_b * self.user_latent_attr + (
                    1.0 - self.W_w_a - self.W_w_b) * self.user_emb_new_2

        # save data
        if update_save:
            self.final_user_emb = self.user_emb_new
            self.final_user_attr_emb = attr_new_input

        self.latent_uu_nosig = self.decode_adj(self.user_emb_new)

        self.latent_uu_la_nosig = self.decode_adj(self.user_emb_new_1)

        return self.latent_uu_nosig, [self.latent_uu_la_nosig], attr_new_input


def normalize_sparse_mat(mx):
    """Row-normalize sparse matrix"""
    # rowsum = torch.sum(mx, 0)
    rowsum = torch.sparse.sum(mx, [0])
    r_inv = torch.pow(rowsum, -0.5)
    r_inv = r_inv.to_dense()
    # r_inv[torch.isinf(r_inv)] = 0. #make inf, -inf, nan to 0. #sparse cannot do this operation
    r_inv[torch.isinf(r_inv)] = 0.  # make inf, -inf, nan to 0. #sparse cannot do this operation
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
        (n_users, n_items, item_features, train, valid, test) = pkl.load(f)
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

    train_adj_label_sparse_np = scipy.sparse.coo_matrix((feas['adj_label'][1], feas['adj_label'][0].transpose()),
                                                        shape=(node_num, node_num))  # .toarray()

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

    earlystop = EarlyStop(patience=10)

    # train shape
    pos_weight = float(A_u_u_sp.shape[0] * A_u_u_sp.shape[1] - A_u_u_sp.sum()) / A_u_u_sp.sum()

    pos_weight_uu = np.ones(node_num) * pos_weight
    pos_weight_uu = Tensor(torch.from_numpy(pos_weight_uu).float())
    bce_withlogitloss_uu = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_uu)  # weighted ce use

    pos_weight_a = float(A_u_a_sp.shape[0] * A_u_a_sp.shape[1] - A_u_a_sp.sum()) / A_u_a_sp.sum()

    pos_weight_ua = np.ones(attr_num) * pos_weight_a
    pos_weight_ua = Tensor(torch.from_numpy(pos_weight_ua).float())
    bce_withlogitloss_ua = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ua)  # weighted ce use

    rgcn = RGCN(node_num, attr_num, opt.recon_adj_hid1_dim, opt.recon_adj_hid2_dim, opt.recon_adj_hid3_dim,
                opt.recon_attr_hid1_dim, opt.recon_attr_hid2_dim, opt.recon_attr_hid3_dim, opt.dropout)

    rgcn.init_weight()

    # --- cuda setting
    if cuda:
        rgcn.cuda()

        bce_withlogitloss_uu = bce_withlogitloss_uu.cuda()
        # bce_withlogitloss_ua = bce_withlogitloss_ua.cuda()

    print(rgcn)

    optimizer_recon_all = torch.optim.Adam(rgcn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    emb_np_for_save = None
    all_batch_index = 0

    def sess_run(epoch, A_u_u, A_u_a, x_u_u, x_u_a, update_save=True):
        '''
        for train 
        '''

        latent_uu_nosig, latent_nosig_list, attr_new_input = rgcn(x_u_a=x_u_a, x_u_u=x_u_u, A_u_a=A_u_a, A_u_u=A_u_u,
                                                                  std=opt.std, tao=opt.tao, update_save=update_save)

        return latent_uu_nosig, latent_nosig_list, attr_new_input.data.cpu().numpy()

    def epoch_run(opt, epoch, mode='infer'):
        '''
        for infer 
        '''
        assert (mode == 'train' or mode == 'infer')

        if mode == 'train':

            update_save = True
            rgcn.train()
        elif mode == 'infer':

            update_save = False
            rgcn.eval()

        latent_uu_nosig, latent_nosig_list, attr_new_input = sess_run(epoch, A_u_u_norm, A_u_a, A_u_u_norm, A_u_a,
                                                                      update_save=update_save)

        if mode == 'train':

            loss_recon_user_user = bce_withlogitloss_uu(latent_uu_nosig,
                                                        A_u_u.to_dense())  # (item_batch, user): be careful, use A_u_i not A_u_ib_norm

            loss_recon_all = loss_recon_user_user
            optimizer_recon_all.zero_grad()
            loss_recon_all.backward()
            optimizer_recon_all.step()


        elif mode == 'infer':
            pass
        if mode == 'train':
            gc.collect()
            #

            # if 'loss_recon_user_attr' in locals():
            #     print('%s:[Epoch %d/%d] user_user_loss=%f user_attr_loss=%f ' % (
            #     datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_user_user.data.cpu().numpy(),
            #     loss_recon_user_attr.data.cpu().numpy()))
            # elif 'loss_recon_uu_3' in locals():
            #     print('%s:[Epoch %d/%d] user_user_loss=%f uu_loss=%f,%f,%f ' % (
            #     datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_user_user.data.cpu().numpy(),
            #     loss_recon_uu_0.data.cpu().numpy(), loss_recon_uu_1.data.cpu().numpy(),
            #     loss_recon_uu_2.data.cpu().numpy()))
            #
            # else:
            #     print('%s:[Epoch %d/%d] loss_all=%f ' % (
            #     datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_all.data.cpu().numpy()))
            print('%s:[Epoch %d/%d] loss_all=%f ' % (
                datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_all.data.cpu().numpy()))

            return
        elif mode == 'infer':
            user_emb_np = rgcn.final_user_emb.data.cpu().numpy()
            return user_emb_np, attr_new_input

    gc.collect()

    print('save network to dir:%s' % opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    lp_dostop_first_str = ''

    for epoch in range(opt.n_epochs):

        print('%s:enter epoch' % datetime.datetime.now().isoformat())
        sys.stdout.flush()

        epoch_run(opt, epoch, mode='train')
        # get emb_np
        # save result

        user_emb_np_for_save, attr_new_input = epoch_run(opt, epoch, mode='infer')

        if opt.net_type == 'uu':

            lm_test = linkpred_metrics(feas, test_edges, test_edges_false)
            emb_np_for_save = user_emb_np_for_save  # TODO:use avg

            test_auc, test_ap, _ = lm_test.get_roc_score(emb_np_for_save, mode='normal')
            #
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
                    ((epoch + 1) % epoch_per == 0) and ((epoch + 1) <= 50)):
                if epoch == 1 and update_best == False:
                    print('just see epoch 1 result:')
                elif epoch + 1 == opt.n_epochs and update_best == False:
                    print('just see final epoch %d result:' % (epoch + 1))
                elif update_best == False:
                    pass
                else:
                    print('update_best: epoch %d' % (epoch + 1))
                if (((epoch + 1) % epoch_per == 0) and ((epoch + 1) <= 50)):
                    print('for nclu, record the emb')

                fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch + 1)
                write_emb_to_pkl_file(emb_np_for_save, fpath_epoch)

                if opt.do_nclu == 1:  # get the result
                    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(emb_np_for_save)
                    print("Epoch:", '%04d' % (epoch + 1))
                    predict_labels = kmeans.predict(emb_np_for_save)
                    cm = clustering_metrics(feas, feas['true_labels'], predict_labels)
                    cm.evaluationClusterModelFromLabel()

        sys.stdout.flush()

    print('%s:writing to  %s and .attr and .attr_new' % (datetime.datetime.now().isoformat(), opt.out_emb_fpath))
    if not os.path.exists(os.path.dirname(opt.out_emb_fpath)):
        os.mkdir(os.path.dirname(opt.out_emb_fpath))

    epoch = opt.n_epochs

    user_emb_np_for_save, attr_new_input = epoch_run(opt, epoch, mode='infer')
    emb_np_for_save = user_emb_np_for_save

    fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch)
    write_emb_to_pkl_file(emb_np_for_save, fpath_epoch)
    if opt.do_nclu == 1:  # get the result
        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(emb_np_for_save)
        print("Epoch:", '%04d' % (epoch + 1))
        predict_labels = kmeans.predict(emb_np_for_save)
        cm = clustering_metrics(feas, feas['true_labels'], predict_labels)
        cm.evaluationClusterModelFromLabel()
    print('over')

    print('lp result:%s' % lp_dostop_first_str)


if __name__ == '__main__':

    # ----command parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--methodname', type=str, required=True, help='methodname ')  #
    parser.add_argument('--save_dir', type=str, required=True, help='save model parameters dir: eg. ../output/xxx/ ')  #
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name: e.g. cora ')  #

    parser.add_argument('--net_type', type=str, required=True, help='net_type: e.g. uu ')  #

    parser.add_argument('--dataset_fea_pkl', type=str, required=True,
                        help='dataset_fea_pkl: e.g.  ../data/xxx/input.pkl')  #

    parser.add_argument('--out_emb_fpath', type=str, required=True,
                        help='out emb fpath: e.g. ../output/vxx/vxxxx.emb ')  #

    # network parameters

    parser.add_argument('--recon_adj_hid1_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 1 dimension')
    parser.add_argument('--recon_adj_hid2_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 2 dimension')
    parser.add_argument('--recon_adj_hid3_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 3 dimension')

    parser.add_argument('--recon_attr_hid1_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 1 dimension')
    parser.add_argument('--recon_attr_hid2_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 2 dimension')
    parser.add_argument('--recon_attr_hid3_dim', type=int, required=True, default=100,
                        help='dimensionality of hidden 3 dimension')

    parser.add_argument('--std', type=float, required=True, default=0.1, help='noise stddev')
    parser.add_argument('--dropout', type=float, required=True, default=0.0, help='dropout value')

    parser.add_argument('--tao', type=float, required=True, default=0.1, help='gumbel softmax temperature')

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
