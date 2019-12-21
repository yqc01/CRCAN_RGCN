#!/bin/env python
# -*- encoding:utf-8 -*-

import os
import sys

import input_data
# import data_helper
from data_helper import *

import pickle as pkl

import heapq
import numpy as np

import argparse
import math
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import random

from sklearn.metrics import f1_score

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
# from sklearn import metrics


'''
use user+item embedding to do prediction
'''


class user_item_pred_metrics():
    # def __init__(self, user_emb, item_emb, train_ui_dict, test_ui_dict):
    def __init__(self, user_emb, item_emb, train_ui_dict, test_ui_dict, testRatings=None, testNegatives=None):
        self.user_emb = user_emb
        self.item_emb = item_emb

        self.train_ui_dict = train_ui_dict
        self.test_ui_dict = test_ui_dict

        self.testRatings = testRatings
        self.testNegatives = testNegatives

    def get_pred_items(self, user):
        pred_value = np.sum(self.user_emb[user] * self.item_emb, axis=1)
        pred_item_value = [(i, v) for i, v in enumerate(pred_value)]
        sorted_pred_item_value = sorted(pred_item_value, key=lambda x: x[1], reverse=True)
        return sorted_pred_item_value

    def get_pred_items_ori_order(self, user):
        pred_value = np.sum(self.user_emb[user] * self.item_emb, axis=1)
        pred_item_value = [(i, v) for i, v in enumerate(pred_value)]
        return pred_item_value

    def eval_mcrec_metric(self, k):
        """
        follow MCRec baseline's eval method, do not remove train data, use 100 neg sample to test value
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        ps, rs, ndcgs = [], [], []

        for idx in range(len(self.testRatings)):
            map_item_score = {}

            rating = self.testRatings[idx]
            items = self.testNegatives[idx]
            u = rating[0]
            gtItems = rating[1:]
            items += gtItems

            predictions = self.get_pred_items_ori_order(u)  # tuple list (user, pred_value)
            for item in items:
                map_item_score[item] = predictions[item][1]

            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            p = self.getP(ranklist, gtItems)
            r = self.getR(ranklist, gtItems)
            ndcg = self.getNDCG(ranklist, gtItems)

            ps.append(p)
            rs.append(r)
            ndcgs.append(ndcg)
        return ps, rs, ndcgs

    def eval(self, k):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        recalls = []

        for user, item_list in self.test_ui_dict.items():
            pred_top_item = self.get_pred_items(user)

            train_set = self.train_ui_dict.get(user, set())  # self.user_to_train_set.get(user_id, set())
            test_set = set(item_list)
            top_n_items = 0
            hits = 0
            for (i, val) in pred_top_item:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    hits += 1
                top_n_items += 1
                if top_n_items == k:
                    break
            recalls.append(hits / float(len(test_set)))
        return recalls

    def getP(self, ranklist, gtItems):
        p = 0
        for item in ranklist:
            if item in gtItems:
                p += 1
        return p * 1.0 / len(ranklist)

    def getR(self, ranklist, gtItems):
        r = 0
        for item in ranklist:
            if item in gtItems:
                r += 1
        return r * 1.0 / len(gtItems)

    def getDCG(self, ranklist, gtItems):
        dcg = 0.0
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item in gtItems:
                dcg += 1.0 / math.log(i + 2)
        return dcg

    def getIDCG(self, ranklist, gtItems):
        idcg = 0.0
        i = 0
        for item in ranklist:
            if item in gtItems:
                idcg += 1.0 / math.log(i + 2)
                i += 1
        return idcg

    def getNDCG(self, ranklist, gtItems):
        dcg = self.getDCG(ranklist, gtItems)
        idcg = self.getIDCG(ranklist, gtItems)
        if idcg == 0:
            return 0
        return dcg / idcg


def main(opt):
    pass
