#!/bin/env python
#-*- encoding:utf-8 -*-

import os
import sys

import input_data
from data_helper import *
from link_pred_with_emb import linkpred_metrics
from user_item_pred import user_item_pred_metrics

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

rgcn_ui

'''

class Recon_all(nn.Module):
    '''

    '''
    def __init__(self, user_num, item_num, user_attr_num, item_attr_num, recon_adj_hid1_dim, recon_adj_hid2_dim, recon_adj_hid3_dim, recon_attr_hid1_dim, recon_attr_hid2_dim , recon_attr_hid3_dim, dropout):
        super(Recon_all, self).__init__()
        self.user_num = user_num
    
        self.user_attr_num = user_attr_num

        self.item_num = item_num
        self.item_attr_num = item_attr_num

        self.recon_adj_hid1_dim = recon_adj_hid1_dim
        self.recon_adj_hid2_dim = recon_adj_hid2_dim
        self.recon_adj_hid3_dim = recon_adj_hid3_dim

        self.recon_attr_hid1_dim = recon_attr_hid1_dim
        self.recon_attr_hid2_dim = recon_attr_hid2_dim
        self.recon_attr_hid3_dim = recon_attr_hid3_dim

        self.dropout = dropout
        

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        


        self.W_w_ulu_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ulu_b = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ili_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ili_b = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ualua_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ialia_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_uani_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_iani_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_uen_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_uen_b = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ien_a = nn.Parameter(torch.empty((1), requires_grad=True))
        self.W_w_ien_b = nn.Parameter(torch.empty((1), requires_grad=True))

        #user
        self.Wui_iia = nn.Parameter(torch.empty((self.item_attr_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wuua_uau = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wui_iu = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))
        
        #item
         
        self.Wiu_uua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wiia_iai = nn.Parameter(torch.empty((self.item_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wiu_ui = nn.Parameter(torch.empty((self.item_num, self.recon_attr_hid2_dim), requires_grad=True))

        #user_attr
        self.Wuau_ui = nn.Parameter(torch.empty((self.item_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wuau_uua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_attr_hid2_dim), requires_grad=True))

        #item_attr
        self.Wiai_iu = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim), requires_grad=True))
        self.Wiai_iia = nn.Parameter(torch.empty((self.item_attr_num, self.recon_attr_hid2_dim), requires_grad=True))


        self.H_attention_u = nn.Parameter(torch.empty((self.user_num, self.recon_attr_hid2_dim, 1), requires_grad=True)) 
        self.H_attention_i = nn.Parameter(torch.empty((self.item_num, self.recon_attr_hid2_dim, 1), requires_grad=True)) 
        self.H_attention_ua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_attr_hid2_dim, 1), requires_grad=True)) 
        self.H_attention_ia = nn.Parameter(torch.empty((self.item_attr_num, self.recon_attr_hid2_dim, 1), requires_grad=True)) 


        self.W_user_aa = nn.Parameter(torch.empty((self.recon_attr_hid2_dim, self.recon_attr_hid2_dim), requires_grad=True)) 
        self.W_item_aa = nn.Parameter(torch.empty((self.recon_attr_hid2_dim, self.recon_attr_hid2_dim), requires_grad=True)) 
        self.W_ua_new = nn.Parameter(torch.empty((self.recon_attr_hid2_dim, self.recon_adj_hid2_dim), requires_grad=True)) 
        self.W_ia_new = nn.Parameter(torch.empty((self.recon_attr_hid2_dim, self.recon_adj_hid2_dim), requires_grad=True)) 
        
        self.linear_ua_new = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim) 
        self.linear_ia_new = torch.nn.Linear(self.recon_attr_hid2_dim, self.recon_attr_hid2_dim) 
        
        #for attention

        #W_ori_uuua is for fixed attr to test
        self.W_ori_ui_iia = nn.Parameter(torch.empty((self.item_attr_num, self.recon_adj_hid2_dim), requires_grad=True)) 
        self.W_ori_iu_uua = nn.Parameter(torch.empty((self.user_attr_num, self.recon_adj_hid2_dim), requires_grad=True)) 

        self.final_user_emb = torch.empty((self.user_num, self.recon_adj_hid2_dim), requires_grad=False) #  
        self.final_user_attr_emb = torch.empty((self.user_attr_num, self.recon_adj_hid2_dim), requires_grad=False) #  
        self.final_item_emb = torch.empty((self.item_num, self.recon_adj_hid2_dim), requires_grad=False) #  
        self.final_item_attr_emb = torch.empty((self.item_attr_num, self.recon_adj_hid2_dim), requires_grad=False) #  



    def init_weight(self):

        torch.nn.init.xavier_uniform_(self.Wui_iia, gain=1)
        torch.nn.init.xavier_uniform_(self.Wuua_uau, gain=1)
        torch.nn.init.xavier_uniform_(self.Wui_iu, gain=1)

        torch.nn.init.xavier_uniform_(self.Wiu_uua, gain=1)
        torch.nn.init.xavier_uniform_(self.Wiia_iai, gain=1)
        torch.nn.init.xavier_uniform_(self.Wiu_ui, gain=1)
        
        torch.nn.init.xavier_uniform_(self.Wuau_ui, gain=1)
        torch.nn.init.xavier_uniform_(self.Wuau_uua, gain=1)
        
        torch.nn.init.xavier_uniform_(self.Wiai_iu, gain=1)
        torch.nn.init.xavier_uniform_(self.Wiai_iia, gain=1)
        
        torch.nn.init.xavier_uniform_(self.H_attention_u, gain=1)
        torch.nn.init.xavier_uniform_(self.H_attention_i, gain=1)

        torch.nn.init.xavier_uniform_(self.H_attention_ua, gain=1)
        torch.nn.init.xavier_uniform_(self.H_attention_ia, gain=1)


        torch.nn.init.xavier_uniform_(self.linear_ua_new.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.linear_ia_new.weight, gain=1)


        torch.nn.init.xavier_uniform_(self.W_ua_new, gain=1)
        torch.nn.init.xavier_uniform_(self.W_ia_new, gain=1)

        torch.nn.init.xavier_uniform_(self.W_user_aa, gain=1)
        torch.nn.init.xavier_uniform_(self.W_item_aa, gain=1)

        torch.nn.init.xavier_uniform_(self.W_ori_ui_iia, gain=1)
        torch.nn.init.xavier_uniform_(self.W_ori_iu_uua, gain=1)


        torch.nn.init.xavier_uniform_(self.final_user_emb, gain=1)
        torch.nn.init.xavier_uniform_(self.final_item_emb, gain=1)
        torch.nn.init.xavier_uniform_(self.final_user_attr_emb, gain=1)
        torch.nn.init.xavier_uniform_(self.final_item_attr_emb, gain=1)
        

        torch.nn.init.constant_(self.W_w_ulu_a, 0.33)
        torch.nn.init.constant_(self.W_w_ulu_b, 0.33)
        torch.nn.init.constant_(self.W_w_ili_a, 0.33)
        torch.nn.init.constant_(self.W_w_ili_b, 0.33)
        torch.nn.init.constant_(self.W_w_ualua_a, 0.5)
        torch.nn.init.constant_(self.W_w_ialia_a, 0.5)
        torch.nn.init.constant_(self.W_w_uani_a, 0.5)
        torch.nn.init.constant_(self.W_w_iani_a, 0.5)
        torch.nn.init.constant_(self.W_w_uen_a, 0.33)
        torch.nn.init.constant_(self.W_w_uen_b, 0.33)
        torch.nn.init.constant_(self.W_w_ien_a, 0.33)
        torch.nn.init.constant_(self.W_w_ien_b, 0.33)

    def graphconv_sparse(self, A, inputs, W, act):
        x = torch.nn.functional.dropout(inputs,p=self.dropout,training=self.training)
        x = torch.mm(x, W) #(node_num, W.shape[1])
        return act(torch.mm(A, x))

    def graphconv(self, A, inputs, W, act):
        x = torch.nn.functional.dropout(inputs,p=self.dropout,training=self.training)
        x = torch.mm(x, W) #(node_num, W.shape[1])
        return act(torch.mm(A, x))

    def decode_adj(self, emb):
        latent_adj_nosig = torch.mm(emb,emb.transpose(1,0))

        return latent_adj_nosig 
    
    def decode_ui(self, u_emb, i_emb):
        latent_ui_nosig = torch.mm(u_emb, i_emb.transpose(1,0))
    
        return latent_ui_nosig 

    def attention_part(self, vec, h, linear=None):
        attn = torch.bmm(vec.detach(), h)
        attention = F.softmax(attn.view(-1, vec.size(1)), dim=1).view(vec.size(0), 1, vec.size(1))
        output_1 = torch.bmm(attention, vec.detach()).view(vec.size(0), vec.size(2))
        if linear is not None:
            output = linear(output_1)
        else:
            output = output_1
        return output, attention

    def norm_mat(self, m, eps):
        return m / (eps + m.norm(p=2, dim=1, keepdim=True).expand_as(m)) 
    def forward(self, x_u_ua, x_u_i, x_i_ia, A_u_ua, A_u_i, A_i_ia, std, tao, update_save=True):
        '''


        '''
        tmp_eps = 1e-16        

        self.user_emb_ui_iia = self.graphconv_sparse(A_u_i, x_i_ia, self.Wui_iia, lambda x:x)
        self.user_emb_ui_iu = self.graphconv_sparse(A_u_i, x_u_i.t(), self.Wui_iu, lambda x:x) 
        self.user_emb_uua_uau = self.graphconv_sparse(A_u_ua, x_u_ua.t(), self.Wuua_uau, lambda x:x) 
        self.user_emb_uua_uau = self.norm_mat(self.user_emb_uua_uau, tmp_eps)

        self.item_emb_iu_uua = self.graphconv_sparse(A_u_i.t(), x_u_ua, self.Wiu_uua, lambda x:x) 
        self.item_emb_iu_ui = self.graphconv_sparse(A_u_i.t(), x_u_i, self.Wiu_ui, lambda x:x) 
        self.item_emb_iia_iai = self.graphconv_sparse(A_i_ia, x_i_ia.t(), self.Wiia_iai, lambda x:x) 
        self.item_emb_iia_iai = self.norm_mat(self.item_emb_iia_iai, tmp_eps) 
        
        self.ia_emb_iai_iia = self.graphconv_sparse(A_i_ia.t(), x_i_ia, self.Wiai_iia, lambda x:x) 
        self.ia_emb_iai_iia = self.norm_mat(self.ia_emb_iai_iia, tmp_eps)
        self.ia_emb_iai_iu = self.graphconv_sparse(A_i_ia.t(), x_u_i.t(), self.Wiai_iu, lambda x:x) 
        self.ia_emb_iai_iu = self.norm_mat(self.ia_emb_iai_iu, tmp_eps)

        self.ua_emb_uau_uua = self.graphconv_sparse(A_u_ua.t(), x_u_ua, self.Wuau_uua, lambda x:x) 
        self.ua_emb_uau_uua = self.norm_mat(self.ua_emb_uau_uua, tmp_eps)
        self.ua_emb_uau_ui = self.graphconv_sparse(A_u_ua.t(), x_u_i, self.Wuau_ui, lambda x:x) 
        self.ua_emb_uau_ui = self.norm_mat(self.ua_emb_uau_ui, tmp_eps)

        #---combine emb
        self.u_latent_ua = self.W_w_ulu_a * self.user_emb_ui_iia + self.W_w_ulu_b * self.user_emb_ui_iu + (1.0 - self.W_w_ulu_a - self.W_w_ulu_b) * self.user_emb_uua_uau


        self.i_latent_ia = self.W_w_ili_a * self.item_emb_iu_uua + self.W_w_ili_b * self.item_emb_iu_ui + (1.0 -
self.W_w_ili_a - self.W_w_ili_b) * self.item_emb_iia_iai

        self.ua_latent_ua = self.W_w_ualua_a * self.ua_emb_uau_uua + (1.0 - self.W_w_ualua_a) * self.ua_emb_uau_ui

        self.ia_latent_ia = self.W_w_ialia_a * self.ia_emb_iai_iia + (1.0 - self.W_w_ialia_a) * self.ia_emb_iai_iu


        self.ua_attr = self.graphconv_sparse(A_u_ua, self.ua_latent_ua, self.W_user_aa, lambda x:x) #user's attr's attr
        self.ua_attr = self.norm_mat(self.ua_attr, tmp_eps)
        self.ia_attr = self.graphconv_sparse(A_i_ia, self.ia_latent_ia, self.W_item_aa, lambda x:x) #user's attr's attr
        self.ia_attr = self.norm_mat(self.ia_attr, tmp_eps)
        
        ua_new_input = self.W_w_uani_a * self.u_latent_ua + (1.0 - self.W_w_uani_a) * self.ua_attr
        ia_new_input = self.W_w_iani_a * self.i_latent_ia + (1.0 - self.W_w_iani_a) * self.ia_attr

        #--new attr
        self.user_emb_new_1 = self.graphconv_sparse(A_u_i, ia_new_input, self.W_ia_new, lambda x:x) #TODO:activation needs changing: sparse
        self.user_emb_new_1 = self.norm_mat(self.user_emb_new_1, tmp_eps)

        self.user_emb_new_2 = self.graphconv_sparse(A_u_i, x_i_ia, self.W_ori_ui_iia, lambda x:x)

        self.user_emb_new = self.W_w_uen_a * self.user_emb_new_1 + self.W_w_uen_b * self.u_latent_ua + (1.0 - self.W_w_uen_a - self.W_w_uen_b) * self.user_emb_new_2


        self.item_emb_new_1 = self.graphconv_sparse(A_u_i.t(), ua_new_input, self.W_ua_new, lambda x:x) 
        self.item_emb_new_1 = self.norm_mat(self.item_emb_new_1, tmp_eps)
        
        self.item_emb_new_2 = self.graphconv_sparse(A_u_i.t(), x_u_ua, self.W_ori_iu_uua, lambda x:x)
        self.item_emb_new = self.W_w_ien_a * self.item_emb_new_1 + self.W_w_ien_b * self.i_latent_ia + (1.0 - self.W_w_ien_a - self.W_w_ien_b) * self.item_emb_new_2

        #save data 
        if update_save:
            self.final_user_emb = self.user_emb_new
            self.final_item_emb = self.item_emb_new

        self.latent_ui_nosig = self.decode_ui(self.user_emb_new, self.item_emb_new)

        
        return self.latent_ui_nosig, (ua_new_input, ia_new_input) 




def normalize_sparse_mat(mx):
    """Row-normalize sparse matrix"""

    rowsum = torch.sparse.sum(mx, [0])
    r_inv = torch.pow(rowsum, -0.5)
    r_inv = r_inv.to_dense()
    r_inv[torch.isinf(r_inv)] = 0. #make inf, -inf, nan to 0. # sparse cannot do this operation
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
    values = dense[indices[0], indices[1]] # modify this based on dimensionality
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
        start_cuda = torch.cuda.FloatTensor([20,18])  #https://github.com/pytorch/pytorch/issues/8856

        T = torch.cuda.sparse.FloatTensor(indices, values, shape)
    else:
        T = torch.sparse.FloatTensor(indices, values, shape)

    return T

def weights_init_normal(m):
    classname = m.__class__.__name__
    print('init_normal:',classname)
    if classname.find('W') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def batch_generator(batch_size,node_list,graph_dict,all_fea_dict,node_num,attr_num):
    cur_batch = [[],[],[]]
    for node in node_list:
        node_adj_index_list = graph_dict[node]
        node_adj_np = np.zeros(node_num, dtype=np.float32)
        node_adj_np[node_adj_index_list] = 1

        node_attr_index_list = all_fea_dict[node]
        node_attr_np = np.zeros(attr_num, dtype=np.float32)
        node_attr_np[node_attr_index_list] = 1

        if len(cur_batch[0]) == batch_size:
            cur_batch = map(lambda x:np.array(x,dtype=np.int64), cur_batch)
            yield cur_batch
            cur_batch = [[], [], []]
        else:
            cur_batch[0].append(node)
            cur_batch[1].append(node_adj_np)
            cur_batch[2].append(node_attr_np)

    if len(cur_batch[0]) > 0:

        cur_batch = map(lambda x:np.array(x,dtype=np.int64), cur_batch)

        yield cur_batch

class EarlyStop(object):
    def __init__(self, patience):
        super(EarlyStop,self).__init__()
        self.patience = patience
        self.best = None
        self.best_epoch = 0
        self.counter = 0
        self.extra_info = None #record the best valid's test score

    def step(self, metric, extra_info, cur_epoch):
        dostop = False
        update_best = False
        if self.best_epoch == 0:

            self.best = metric
            self.best_epoch = cur_epoch
            self.extra_info = extra_info
            update_best = True
        elif metric < self.best:
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


        return dostop, update_best

def read_ui_train_test_pkl(fpath):

    with open(fpath, 'rb') as f:
        (n_users, n_items, item_features, train, valid, test) = pkl.load(f) #here features is not the returned features
    #dok_matrix
    return n_users, n_items, item_features, train, valid, test 

def main(opt, cuda, Tensor, LongTensor, ByteTensor, margin, dataset_fea_pkl_fpath):
    #read train data (link prediction task)

    if opt.net_type == 'ui':
        n_users, n_items, item_features, train, valid, test = read_ui_train_test_pkl(dataset_fea_pkl_fpath)
        n_item_attr = item_features.shape[1]
        n_user_attr = n_items

        assert(n_items == item_features.shape[0])

    i_ia_coo = item_features.tocoo()
    train_coo = train.tocoo()
    A_u_i_sp = train_coo
    A_i_ia_sp = i_ia_coo

    A_u_i = to_torch_sparse_tensor(A_u_i_sp, cuda)
    A_i_ia = to_torch_sparse_tensor(A_i_ia_sp, cuda)

    A_u_i_norm = normalize_sparse_mat(A_u_i)
    A_u_ua = A_u_i
    A_u_ua_norm = A_u_i_norm

    print('user, item, (user_attr), item_attr: %d, %d, %d, %d' % (n_users, n_items, n_user_attr, n_item_attr))

    if opt.net_type == 'ui':
        train_ui_dict = dict()
        test_ui_dict = dict()
        valid_ui_dict = dict()
        for (edge, value) in train.items():
            u, i = edge
            if u not in train_ui_dict:
                train_ui_dict[u] = [i]
            else:
                train_ui_dict[u].append(i)

        for (edge, value) in valid.items():
            u, i = edge
            if u not in valid_ui_dict:
                valid_ui_dict[u] = [i]
            else:
                valid_ui_dict[u].append(i)

        for (edge, value) in test.items():
            u, i = edge
            if u not in test_ui_dict:
                test_ui_dict[u] = [i]
            else:
                test_ui_dict[u].append(i)

    earlystop = EarlyStop(patience = 20)


    #train shape
    pos_weight = float(A_u_i_sp.shape[0] * A_u_i_sp.shape[1] - A_u_i_sp.sum()) / A_u_i_sp.sum()

    pos_weight_ui = np.ones(n_items) * pos_weight
    pos_weight_ui = Tensor(torch.from_numpy(pos_weight_ui).float())
    bce_withlogitloss_ui = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_ui) #weighted ce use


    rgcn_ui = Recon_all(n_users, n_items, n_user_attr, n_item_attr, opt.recon_adj_hid1_dim, opt.recon_adj_hid2_dim, opt.recon_adj_hid3_dim, opt.recon_attr_hid1_dim, opt.recon_attr_hid2_dim, opt.recon_attr_hid3_dim, opt.dropout)

    rgcn_ui.init_weight()


    #--- cuda setting
    if cuda:
        rgcn_ui.cuda()

        bce_withlogitloss_ui = bce_withlogitloss_ui.cuda()

    print(rgcn_ui)


    optimizer_recon_all = torch.optim.Adam(rgcn_ui.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    emb_np_for_save = None
    all_batch_index = 0
    

    def sess_run(epoch, A_u_i, A_u_ua, A_i_ia, x_u_i, x_u_ua, x_i_ia, update_save=True):
        '''
        for train 
        '''

       
        latent_ui_nosig, (ua_new_input, ia_new_input) = rgcn_ui(x_u_ua=x_u_ua, x_i_ia=x_i_ia, x_u_i=x_u_i, A_u_ua=A_u_ua, A_i_ia=A_i_ia, A_u_i=A_u_i, std=opt.std, tao=opt.tao, update_save=update_save)


        return latent_ui_nosig, ia_new_input.data.cpu().numpy()

    def generate_new_attr_from_list(opt, attr_new_input):
        attr_new_input_all = []
        item_index_list = list(range(0, n_items))  
        attr_new_input_all.extend(add_new_attr(attr_new_input, A_a_i_sp.toarray().transpose(), sample_attr_num, item_index_list))
        return attr_new_input_all

    def epoch_run(opt, epoch, mode='infer'):
        '''
        for infer 
        '''
        assert(mode == 'train' or mode == 'infer')

        if mode == 'train':

            update_save = True
            rgcn_ui.train()
        elif mode == 'infer':

            update_save = False
            rgcn_ui.eval()



        latent_ui_nosig, ia_new_input = sess_run(epoch=epoch, A_u_i=A_u_i_norm, A_u_ua=A_u_ua, A_i_ia=A_i_ia, x_u_i=A_u_i_norm, x_u_ua=A_u_ua, x_i_ia=A_i_ia, update_save=update_save) 

        if mode == 'train':

            loss_recon_user_item = bce_withlogitloss_ui(latent_ui_nosig, A_u_i.to_dense()) #(item_batch, user): be careful, use A_u_i not A_u_ib_norm
            loss_recon_all = loss_recon_user_item
            optimizer_recon_all.zero_grad()
            loss_recon_all.backward()
            optimizer_recon_all.step()
                    
                
        elif mode == 'infer':
            pass
        if mode == 'train':
            gc.collect()
#

            if 'loss_recon_user_attr' in locals():
                print('%s:[Epoch %d/%d] user_user_loss=%f user_attr_loss=%f ' % (datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_user_user.data.cpu().numpy(), loss_recon_user_attr.data.cpu().numpy()  ))
            else:
                print('%s:[Epoch %d/%d] loss_all=%f ' % (datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, loss_recon_all.data.cpu().numpy() ))

            return 
        elif mode == 'infer':
            user_emb_np = rgcn_ui.final_user_emb.data.cpu().numpy()
            item_emb_np = rgcn_ui.final_item_emb.data.cpu().numpy()

            return user_emb_np, item_emb_np, ia_new_input

    gc.collect()

    print('save network to dir:%s' % opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    for epoch in range(opt.n_epochs):

        print('%s:enter epoch' % datetime.datetime.now().isoformat()) 
        sys.stdout.flush()

        
        epoch_run(opt, epoch, mode='train') 
        #get emb_np
        #save result
            
        user_emb_np_for_save, item_emb_np_for_save, ia_new_input = epoch_run(opt, epoch, mode='infer')

        if opt.net_type == 'ui':
            ui_pred_test = user_item_pred_metrics(user_emb_np_for_save, item_emb_np_for_save, train_ui_dict, test_ui_dict)           
            test_recall = np.mean(ui_pred_test.eval(opt.recall_k))

            print('%s:[Epoch %d/%d] test recall@%d:%f' % (datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, opt.recall_k, test_recall))
            ui_pred_valid = user_item_pred_metrics(user_emb_np_for_save, item_emb_np_for_save, train_ui_dict, valid_ui_dict)           
            valid_recall = np.mean(ui_pred_valid.eval(opt.recall_k))
            print('%s:[Epoch %d/%d] valid recall@%d:%f' % (datetime.datetime.now().isoformat(), epoch + 1, opt.n_epochs, opt.recall_k, valid_recall))
            extra_info = 'test recall:%f' %(test_recall)

            update_best = False
            dostop = False
            if epoch+1 >= 50:
                dostop, update_best = earlystop.step(valid_recall, extra_info, epoch + 1)

                if dostop == True :
                    print("dostop at epoch %d | valid recall:%f, best val recall:%f at epoch :%d | best %s" %(epoch+1, valid_recall, earlystop.best, earlystop.best_epoch, earlystop.extra_info))

            if update_best == True or (epoch+1) == 1:
                if (epoch+1) == 1:
                    print('just test writing epoch')
                else:
                    print('update recall at epoch %d: %s' % (epoch + 1, extra_info))

                fpath_epoch = '%s.ep%d' % (opt.out_emb_fpath, epoch+1)

                write_ui_emb_to_pkl_file(user_emb_np_for_save, item_emb_np_for_save, fpath_epoch)


        sys.stdout.flush()

    print('%s:writing to  %s and .attr and .attr_new' % (datetime.datetime.now().isoformat(),opt.out_emb_fpath))
    if not os.path.exists(os.path.dirname(opt.out_emb_fpath)):
        os.mkdir(os.path.dirname(opt.out_emb_fpath))


    epoch = opt.n_epochs


    print('over')


if __name__ == '__main__':

    #----command parameters
    parser = argparse.ArgumentParser()


    parser.add_argument('--methodname', type=str, required=True, help='methodname ') #
    parser.add_argument('--save_dir', type=str, required=True, help='save model parameters dir: eg. ../output/vxx/ ') #
    parser.add_argument('--dataset_name', type=str, required=True,  help='dataset name: e.g. cora ') #
    
    parser.add_argument('--net_type', type=str, required=True, help='net_type: ui') #

    parser.add_argument('--dataset_fea_pkl', type=str, required=True,  help='dataset_fea_pkl: e.g.  ../data/useritem/citeulike/input.pkl') #
    
    parser.add_argument('--out_emb_fpath', type=str, required=True, help='out emb fpath: e.g. ../output/vxx/vxxxx.emb ') #


    #network parameters


    parser.add_argument('--recon_adj_hid1_dim', type=int, required=True, default=100, help='dimensionality of hidden 1 dimension') #TODO:set dim value later
    parser.add_argument('--recon_adj_hid2_dim', type=int, required=True, default=100, help='dimensionality of hidden 2 dimension') #TODO:set dim value later
    parser.add_argument('--recon_adj_hid3_dim', type=int, required=True, default=100, help='dimensionality of hidden 3 dimension') #TODO:set dim value later

    parser.add_argument('--recon_attr_hid1_dim', type=int, required=True, default=100, help='dimensionality of hidden 1 dimension') #TODO:set dim value later
    parser.add_argument('--recon_attr_hid2_dim', type=int, required=True, default=100, help='dimensionality of hidden 2 dimension') #TODO:set dim value later
    parser.add_argument('--recon_attr_hid3_dim', type=int, required=True, default=100, help='dimensionality of hidden 3 dimension') #TODO:set dim value later

    parser.add_argument('--std', type=float, required=True, default=0.1, help='noise stddev')
    parser.add_argument('--dropout', type=float, required=True, default=0.0, help='dropout value')

    parser.add_argument('--tao', type=float, required=True, default=0.1, help='gumbel softmax temperature')


    #adam para
    parser.add_argument('--lr', type=float, required=True, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, required=True, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, required=True, default=0.999, help='adam: decay of first order momentum of gradient')

    #learn para
    parser.add_argument('--n_epochs', type=int, required=True, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, required=True, default=1, help='size of the batches') # origin is 64

    parser.add_argument('--recall_k', type=int, required=True, help='recall@k number: e.g. 50 ') #
    opt = parser.parse_args()  #option

    dataset_fea_pkl_fpath = opt.dataset_fea_pkl# '../data/pkl/%s_fea.pkl' % (opt.dataset_name)
    print("arguments:%s" % opt)
    print("read feature(uu / ui, attr) pkl path:%s"% dataset_fea_pkl_fpath)

    margin = max(1, opt.batch_size / 64.)
    cuda = False 

    Tensor = lambda x:torch.cuda.FloatTensor(x.cuda()) if cuda else torch.FloatTensor(x)
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    print("margin:%s"% margin)
    print("cuda:%s"% cuda)
    print("Tensor:%s"% Tensor)
    
    if opt.net_type not in ['uu', 'ui']:
        print('unknown net_type')
        exit()
    #set seed
    seednum = 1
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    random.seed(seednum)

    main(opt,cuda, Tensor, LongTensor, ByteTensor, margin, dataset_fea_pkl_fpath)

