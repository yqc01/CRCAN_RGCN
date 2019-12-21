#!/bin/env python
# -*- encoding:utf-8 -*-
import numpy as np
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict
import os
import sys
import pickle as pkl
from collections import defaultdict
import datetime
import networkx as nx
import scipy.sparse as sp
import scipy
import torch
import copy


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
        start_cuda = torch.cuda.FloatTensor([5, 18])  # https://github.com/pytorch/pytorch/issues/8856

        T = torch.cuda.sparse.FloatTensor(indices, values, shape)
    else:
        T = torch.sparse.FloatTensor(indices, values, shape)

    return T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_train_fit(emb_np, train_edges):
    adj_rec = np.dot(emb_np, emb_np.T)
    best_prec = 0.0
    best_threshold = 0
    for threshold_step in range(1, 101):
        threshold = 1.0 / 100 * threshold_step
        pred_right = 0
        for e in train_edges:
            pred_val = sigmoid(adj_rec[e[0], e[1]])
            if pred_val >= threshold:
                pred_right += 1
        if pred_right > best_prec:
            best_prec = pred_right
            best_threshold = threshold

    return 1.0 * best_prec / len(train_edges), best_threshold


def write_ui_emb_to_pkl_file(emb_u_np, emb_i_np, pkl_out_emb_fpath):
    # use pkl
    with open(pkl_out_emb_fpath, 'wb')as f:
        pkl.dump([emb_u_np, emb_i_np], f, protocol=2)


def write_emb_to_pkl_file(emb_np, pkl_out_emb_fpath):
    # use pkl
    with open(pkl_out_emb_fpath, 'wb')as f:
        pkl.dump(emb_np, f, protocol=2)


def read_emb_from_pkl_file(emb_fpath):
    with open(emb_fpath, 'rb') as f:
        obj = pkl.load(f)
    return obj


def write_emb_to_file(emb_np, out_emb_fpath):
    with open(out_emb_fpath, 'w') as f:
        for i in range(0, len(emb_np)):
            s = '%d\t' % i
            s += '\t'.join(list(map(str, list(emb_np[i]))))
            f.write(s + '\n')


def write_ui_emb_to_file(user_emb_np, item_emb_np, out_emb_fpath):
    with open(out_emb_fpath + '.user', 'w') as f:
        for i in range(0, len(user_emb_np)):
            s = '%d\t' % i
            s += '\t'.join(list(map(str, list(user_emb_np[i]))))
            f.write(s + '\n')

    with open(out_emb_fpath + '.item', 'w') as f:
        for i in range(0, len(item_emb_np)):
            s = '%d\t' % i
            s += '\t'.join(list(map(str, list(item_emb_np[i]))))
            f.write(s + '\n')


def write_item_attr_info_to_file(attr_input_sparse, attr_new_pred_list, original_pkl_data, add_attr_num, out_emb_fpath):
    (n_users, n_items, item_features, train, valid, test) = original_pkl_data

    print('require add attr number for each item: %d' % add_attr_num)
    print('original item_attr nonzero number: %d' % item_features.count_nonzero())

    # create new item features
    item_features_new = sp.dok_matrix(item_features)
    print('prepare to add attr number: %d' % len(attr_new_pred_list))
    for edge_info in attr_new_pred_list:
        edge = (edge_info[0], edge_info[1])
        val = edge_info[2]
        if edge not in item_features_new:
            item_features_new[edge] = val

    print('after adding: item_attr new nonzero number: %d' % item_features_new.count_nonzero())

    out_pkl_fpath = out_emb_fpath + '.enhance.pkl'
    print('writing new pkl to %s' % out_pkl_fpath)
    with open(out_pkl_fpath, 'wb')as f:
        pkl.dump([n_users, n_items, item_features_new, train, valid, test], f, protocol=2)


def write_new_attr_info_to_pkl(original_pkl_data, attr_pred, add_attr_num, out_emb_fpath):
    pkl_data_copy = copy.deepcopy(original_pkl_data)

    node_num = pkl_data_copy['num_nodes']
    attr_num = pkl_data_copy['num_features']
    display_once = 0

    attr_new_pred_list = []
    user_attr_mat = scipy.sparse.coo_matrix((pkl_data_copy['features'][1], pkl_data_copy['features'][0].transpose()),
                                            shape=(node_num, attr_num)).todok()

    attr_true = user_attr_mat.toarray()

    print('require add attr number for each node: %d' % add_attr_num)
    print('original node_attr nonzero number: %d' % user_attr_mat.count_nonzero())
    attr_eps = 1.0 / attr_num  # 1e-10#
    print('attr_eps:%f' % attr_eps)

    added_attr_list = []
    added_attr_value_list = []

    for i in range(len(attr_true)):
        tmp_list = np.array(list(attr_pred[i]))  # TODO:this may be sparse ??
        tmp_list[attr_true[i] > 0] = 0.0
        tmp_attr_tuple = [(attr_index, attr_pred_val) for attr_index, attr_pred_val in enumerate(list(tmp_list))]
        sorted_tmp_attr_tuple = sorted(tmp_attr_tuple, key=lambda x: x[1], reverse=True)

        if display_once < 5:
            print('sorted_tmp_attr_tuple[:10]:', sorted_tmp_attr_tuple[:10])
            display_once += 1

        for (attr_index, attr_pred_val) in sorted_tmp_attr_tuple:
            if attr_pred_val <= attr_eps:
                break
            added_attr_list.append((i, attr_index))
            added_attr_value_list.append(attr_pred_val)
    print('added attr number:%d' % len(added_attr_list))

    added_attr_list.extend(user_attr_mat.keys())
    added_attr_value_list.extend(user_attr_mat.values())
    print('total attr number now:%d' % len(added_attr_list))

    pkl_data_copy['features'] = (
    np.array(added_attr_list, dtype=np.int32), np.array(added_attr_value_list, dtype=np.float32),
    user_attr_mat.shape)  # user_attr_mat.tocsr()
    pkl_data_copy['features_nonzero'] = pkl_data_copy['features'][0].shape[0]

    out_pkl_fpath = out_emb_fpath + '.enhance.pkl'
    print('writing new pkl to %s' % out_pkl_fpath)
    with open(out_pkl_fpath, 'wb')as f:
        pkl.dump(pkl_data_copy, f, protocol=2)


def write_attr_info_to_file(attr_input, attr_new_input, ui_to_ui_index_dict, max_user_index_plus1, out_emb_fpath):
    '''
    ui_to_ui_index_dict: actually user/item name -> user/item index
    max_user_index_plus1: item_name = item_real_name + max_user_index_plus1

    write 'item_name tag1 tag2 ...' to file (original and predicted)
    '''
    ui_index_to_ui_dict = {v: k for k, v in ui_to_ui_index_dict.items()}

    with open(out_emb_fpath + '.attr', 'w') as f:
        for i in range(0, len(attr_input)):
            if i not in ui_index_to_ui_dict:
                continue
            ui_name = ui_index_to_ui_dict[i]
            if ui_name >= max_user_index_plus1:  # this is item
                item_real_name = ui_name - max_user_index_plus1
                s = '%d\t' % item_real_name
                s += '\t'.join(list(map(str, list(attr_input[i]))))
                f.write(s + '\n')

    with open(out_emb_fpath + '.attr_new', 'w') as f:
        for i in range(0, len(attr_new_input)):
            if i not in ui_index_to_ui_dict:
                continue
            ui_name = ui_index_to_ui_dict[i]
            if ui_name >= max_user_index_plus1:  # this is item
                item_real_name = ui_name - max_user_index_plus1
                s = '%d\t' % item_real_name
                s += '\t'.join(list(map(str, list(attr_new_input[i]))))
                f.write(s + '\n')


def format_data_ui_concat(data_name, has_features=1):
    '''
    concat u and i ==> get a u-u mat and its map 
    '''
    # Load data

    fpath_dir = '../data/useritem/%s/' % data_name
    fpath_input = '%sinput.pkl' % fpath_dir
    with open(fpath_input, 'rb') as f:
        (n_users, n_items, item_features, train, valid, test) = pkl.load(
            f)  # here features is not the returned features
    ui_graph = defaultdict(list)
    ii_graph = defaultdict(set)
    ii_graph_list = defaultdict(list)  # dict()

    user_set = set()
    item_set = set()
    tag_set = set()

    for edge, value in train.items():
        u, i = edge
        user_set.add(u)
        item_set.add(i)

    for edge, value in valid.items():
        u, i = edge
        user_set.add(u)
        item_set.add(i)

    for edge, value in test.items():
        u, i = edge
        user_set.add(u)
        item_set.add(i)

    # check if n_users is from [0, n_users-1]
    print(n_users)
    print('user:len, min, max:', len(user_set), min(user_set), max(user_set))
    print(n_items)
    print('item:len, min, max:', len(item_set), min(item_set), max(item_set))

    max_user_index_plus1 = max(user_set) + 1
    user_plus_item_num = (max_user_index_plus1 + max(item_set)) + 1
    new_ui_edge_dict = defaultdict(list)

    for edge, value in train.items():
        u, i = edge
        new_ui_edge_dict[u].append(i + max_user_index_plus1)
        new_ui_edge_dict[i + max_user_index_plus1].append(u)

    print('%s:get ii mat' % (datetime.datetime.now().isoformat()))
    G_ui = nx.from_dict_of_lists(new_ui_edge_dict)
    G_ui_nodes_list = list(G_ui.nodes())
    adj = nx.adjacency_matrix(G_ui)
    ui_to_ui_index_dict = dict()  # include u and i
    for i in range(len(G_ui_nodes_list)):
        ui_to_ui_index_dict[G_ui_nodes_list[i]] = i

    print('adj shape:', adj.get_shape())

    tag_set = set()
    for (item, tag), value in item_features.items():
        tag_set.add(tag)
    max_tag_num = max(tag_set) + 1
    item_features_mapped = sp.dok_matrix((user_plus_item_num, max_tag_num), dtype=np.int64)

    unused_item_cnt = 0  # no user item info's item
    for (item, tag), value in item_features.items():
        # not used item
        item_mapped = item + max_user_index_plus1
        if item_mapped not in ui_to_ui_index_dict:
            unused_item_cnt += 1
        else:
            item_features_mapped[ui_to_ui_index_dict[item_mapped], tag] = value

    print('unused_item_cnt: %d' % unused_item_cnt)

    features = item_features_mapped.tolil()  # item_features.tolil()
    # map to new position
    # true_labels: the neighbor truth : not used for me and arga...
    true_labels = None

    # --transform over, now follows the original procedure
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # to remove adj_matirx's diag,  offset is 0
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    print('%s:mask test edges over' % (datetime.datetime.now().isoformat()))
    # if FLAGS.features == 0:
    if has_features == 0:
        features = sp.identity(features.shape[0])  # featureless #just diag have 1

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features,
             true_labels, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig,
             ui_to_ui_index_dict, max_user_index_plus1]  # add ui
    feas = {}
    for item in items:
        feas[retrieve_name(item)] = item

    return feas


def format_data_ui(data_name, has_features=1):
    # Load data

    fpath_dir = '../data/useritem/%s/' % data_name
    fpath_input = '%sinput.pkl' % fpath_dir
    with open(fpath_input, 'rb') as f:
        (n_users, n_items, item_features, train, valid, test) = pkl.load(
            f)  # here features is not the returned features
    ui_graph = defaultdict(list)
    ii_graph = defaultdict(set)
    ii_graph_list = defaultdict(list)  # dict()
    for edge, value in train.items():
        u, i = edge
        ui_graph[u].append(i)
    #
    edge_dict = defaultdict(int)
    tmp_u_number = len(ui_graph)
    for index, (u, ilist) in enumerate(ui_graph.items()):

        if index % 500 == 0:
            print('user number: %d/%d' % (index, tmp_u_number))
        for i in ilist:
            for j in ilist:
                # ii_graph[i].add(j)
                if i != j:
                    edge_dict[(i, j)] += 1
        if len(edge_dict) % 5000 == 0:
            print('len(edge_dict):%d' % len(edge_dict))

    print('len(edge_dict):%d' % len(edge_dict))
    edge_visit_thresh = 2

    for edge, visit_num in edge_dict.items():
        i1, i2 = edge
        if visit_num >= edge_visit_thresh:
            ii_graph_list[i1].append(i2)  # = list(iset)
    print('%s:get ii mat' % (datetime.datetime.now().isoformat()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(ii_graph_list))
    print('adj shape:', adj.get_shape())

    # features: lil_matrix
    features = item_features.tolil()

    # true_labels: the neighbor truth : not used for me and arga...
    true_labels = None

    # --transform over, now follows the original procedure
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    print('%s:mask test edges over' % (datetime.datetime.now().isoformat()))
    if has_features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features,
             true_labels, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig]
    feas = {}
    for item in items:
        feas[retrieve_name(item)] = item

    return feas


def format_data(data_name, has_features=1):
    # Load data
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    if has_features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features,
             true_labels, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item

    return feas


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def read_emb_file(fpath):
    node_num = 0
    latent_dim = 0
    with open(fpath) as f:
        line = f.readline()
        if len(line) > 0:
            items = line.strip().split('\t')
            node_num += 1
            latent_dim = len(items) - 1
        for line in f:
            node_num += 1

    emb_np = np.zeros((node_num, latent_dim), dtype=np.float32)  # [node_num, dim] float value
    with open(fpath) as f:
        for line in f:
            items = line.strip().split('\t')
            uid = int(items[0])
            val = list(map(lambda x: float(x), items[1:]))
            emb_np[uid] = np.array(val)

    return emb_np


def read_train_test_pkl(train_test_pkl_fpath):
    if not os.path.exists(train_test_pkl_fpath):
        print('train data not found , exit(): use prepare_dataset.py to generate train data first')
        exit()
    else:
        print('reading feature pkl path: %s' % train_test_pkl_fpath)
        with open(train_test_pkl_fpath, 'rb') as f:
            feas = pkl.load(f)
    return feas
