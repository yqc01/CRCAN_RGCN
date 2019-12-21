#!/bin/env python
# -*- encoding:utf-8 -*-

import os
import sys

import input_data
# import data_helper
from data_helper import *

import pickle as pkl

import numpy as np

import argparse
import math
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import random

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics

'''
use node embedding to do link prediction
'''


class linkpred_metrics():
    def __init__(self, feas, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg
        self.feas = feas

    def get_roc_score_ui_uu(self, emb_user, emb_item, mode):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def calc_mode(vec):
            ret = sum(map(lambda x: x ** 2, vec))
            return math.sqrt(ret)

        def cos_val(vec1, vec2):
            v1 = sum(map(lambda x, y: x * y, vec1, vec2))
            return v1 / (calc_mode(vec1) * calc_mode(vec2))  # according to its realization: range[-1,1]

        # Predict on test set of edges
        if mode.lower() == 'normal':
            # adj_rec = np.dot(emb, emb.T)
            adj_rec = np.dot(emb_user, emb_item.T)
        elif mode.lower() == 'anrl':

            pass  # since it only calc those test pred
        else:
            print('unknown link pred mode')
            exit()

        preds = []
        pos = []
        for e in self.edges_pos:
            if mode.lower() == 'normal':
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
            elif mode.lower() == 'anrl':
                preds.append(cos_val(emb_user[e[0]], emb_item[e[1]]))
            else:
                print('unknown link pred mode')
                exit()

            pos.append(self.feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            if mode.lower() == 'normal':
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            elif mode.lower() == 'anrl':
                preds_neg.append(cos_val(emb_user[e[0]], emb_item[e[1]]))
            else:
                print('unknown link pred mode')
                exit()
            neg.append(self.feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_roc_score(self, emb, mode):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def calc_mode(vec):
            ret = sum(map(lambda x: x ** 2, vec))
            return math.sqrt(ret)

        def cos_val(vec1, vec2):
            v1 = sum(map(lambda x, y: x * y, vec1, vec2))
            return v1 / (calc_mode(vec1) * calc_mode(vec2))  # according to its realization: range[-1,1]

        # Predict on test set of edges
        if mode.lower() == 'normal':
            adj_rec = np.dot(emb, emb.T)
        elif mode.lower() == 'anrl':

            pass  # since it only calc those test pred
        else:
            print('unknown link pred mode')
            exit()

        preds = []
        pos = []
        for e in self.edges_pos:
            if mode.lower() == 'normal':
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
            elif mode.lower() == 'anrl':
                preds.append(cos_val(emb[e[0]], emb[e[1]]))
            else:
                print('unknown link pred mode')
                exit()

            pos.append(self.feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            if mode.lower() == 'normal':
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            elif mode.lower() == 'anrl':
                preds_neg.append(cos_val(emb[e[0]], emb[e[1]]))
            else:
                print('unknown link pred mode')
                exit()
            neg.append(self.feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, emb


def main(opt):
    feas = read_train_test_pkl(opt.train_test_pkl)

    train_edges = feas['train_edges']  # np array( int32)
    all_feas = feas['features']  # csr matrix (2708,1433)

    val_edges = feas['val_edges']  # np array (int32)
    val_edges_false = feas['val_edges_false']  # edge list

    test_edges = feas['test_edges']  # np array (int32)
    test_edges_false = feas['test_edges_false']  # edge list

    train_graph_dict = dict()
    all_fea_dict = dict()

    if opt.emb_format == 'emb':
        emb_np = read_emb_file(opt.input_emb)
    elif opt.emb_format == 'pkl':
        with open(opt.input_emb, 'rb') as f:
            emb_np = pkl.load(f)
    else:
        print('unknown emb_format')
        exit()

    lm_test = linkpred_metrics(feas, feas['test_edges'], feas['test_edges_false'])
    roc_score, ap_score, _ = lm_test.get_roc_score(emb_np, opt.mode)

    print('Test ROC=%f, Test AP=%f' % (roc_score, ap_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_emb', type=str, required=True,
                        help='node embedding file name: e.g. ../output/vxx/vxxxx.emb')
    parser.add_argument('--train_test_pkl', type=str, required=True,
                        help='train test info file name: e.g. ../data/pkl/<dataset_name>_fea.pkl')

    parser.add_argument('--mode', type=str, required=True, help='link prediction mode: e.g. normal/anrl')  #
    parser.add_argument('--emb_format', type=str, required=True, help='emb_format: e.g. emb/pkl')  #

    opt = parser.parse_args()  # option

    print('arguments:', opt)

    # set seed
    seednum = 2018
    np.random.seed(seednum)
    # torch.manual_seed(seednum)
    random.seed(seednum)

    main(opt)
