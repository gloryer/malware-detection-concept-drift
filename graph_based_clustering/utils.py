
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GINConv, GlobalAvgPool, GCNConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor






################################################################################
# Load data
################################################################################

class GraphData(Dataset):

    def __init__(self, cfg_path, **kwargs):
        self.cfg_path = cfg_path

        super().__init__(**kwargs)

    def read(self):

        file_list = os.listdir(self.cfg_path)
        file_list_x_y = list(filter(lambda x: '_sparse_matrix' not in x and '.npz' in x, file_list))

        # print(len(file_list_x_y))
        output = []

        for filepath in file_list_x_y:

            # full path of node attribute and label
            fullpath = os.path.join(self.cfg_path, filepath)
            # file path of adj matrix
            filepath_sp = filepath.split('.')[0] + "_sparse_matrix.npz"
            # full path pf adj matrix
            fullpath_sp = os.path.join(self.cfg_path, filepath_sp)
            # with open(fullpath_sp, 'rb') as f1:
            sparse_matrix = sp.load_npz(fullpath_sp)
            sparse_matrix = sparse_matrix.astype('float32')

            # with open(fullpath, 'rb') as f2:
            data = np.load(fullpath)

            # Remove diagonal elements
            adj = sparse_matrix - sp.dia_matrix((sparse_matrix.diagonal()[np.newaxis, :], [0]),
                                                shape=sparse_matrix.shape)
            adj.eliminate_zeros()
            # Check that diag is zero:
            assert np.diag(adj.todense()).sum() == 0

            adj_triu = sp.triu(adj)
            adj_tuple = sparse_to_tuple(adj_triu)
            edges = adj_tuple[0]

            #             if edges.shape[0] >= 3:
            #                 i+=1

            #             if i ==124:
            #                 print(edges.shape[0])
            #                 print(edges_all.shape[0])
            #                 print(sparse_matrix.shape)
            #                 print(adj_triu)
            #                 print("\n")
            #                 print(sparse_matrix)
            #                 print("\n")
            #                 print(adj)
            #                 output.append(Graph(x=data['x'], a= sparse_matrix, y=data['y']))
            #                 break
            #                print(edges.shape[0])

            #             if edges.shape[0] ==2:
            #                 print(True)
            #                 print(i)

            # important! filter out noisy data where the number of basic block is less than 10 and the number of non-self edges in the upper triangle is less than 3
            # if the graph is too large, we will run into oom problems
            if data["x"].shape[0] >= 10 and edges.shape[0] >= 3 and sparse_matrix.shape[0] <= 46000:
                output.append(Graph(x=data['x'], a=sparse_matrix, y=data['y']))

        return output


################################################################################
# Load data
################################################################################

class obtain_ID():

    def __init__(self, cfg_path, file_list_x_y):
        self.cfg_path = cfg_path
        self.file_list_x_y = file_list_x_y

        super().__init__()

    def read(self):

        # print(len(file_list_x_y))
        output = []

        for filepath in self.file_list_x_y:

            # full path of node attribute and label
            fullpath = os.path.join(self.cfg_path, filepath)
            # file path of adj matrix
            filepath_sp = filepath.split('.')[0] + "_sparse_matrix.npz"
            # full path pf adj matrix
            fullpath_sp = os.path.join(self.cfg_path, filepath_sp)
            # with open(fullpath_sp, 'rb') as f1:
            sparse_matrix = sp.load_npz(fullpath_sp)
            sparse_matrix = sparse_matrix.astype('float32')

            # with open(fullpath, 'rb') as f2:
            data = np.load(fullpath)

            # Remove diagonal elements
            adj = sparse_matrix - sp.dia_matrix((sparse_matrix.diagonal()[np.newaxis, :], [0]),
                                                shape=sparse_matrix.shape)
            adj.eliminate_zeros()
            # Check that diag is zero:
            assert np.diag(adj.todense()).sum() == 0

            adj_triu = sp.triu(adj)
            adj_tuple = sparse_to_tuple(adj_triu)
            edges = adj_tuple[0]

            #             if edges.shape[0] >= 3:
            #                 i+=1

            #             if i ==124:
            #                 print(edges.shape[0])
            #                 print(edges_all.shape[0])
            #                 print(sparse_matrix.shape)
            #                 print(adj_triu)
            #                 print("\n")
            #                 print(sparse_matrix)
            #                 print("\n")
            #                 print(adj)
            #                 output.append(Graph(x=data['x'], a= sparse_matrix, y=data['y']))
            #                 break
            #                print(edges.shape[0])

            #             if edges.shape[0] ==2:
            #                 print(True)
            #                 print(i)

            # important! filter out noisy data where the number of basic block is less than 10 and the number of non-self edges in the upper triangle is less than 3
            # if the graph is too large, we will run into oom problems
            if data["x"].shape[0] >= 10 and edges.shape[0] >= 3 and sparse_matrix.shape[0] <= 46000:
                output.append(filepath)

        return output







class list_to_spektral_dataset(Dataset):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(**kwargs)

    def read(self):
        # print(self.data[0])
        return [Graph(x=graph.x, a=graph.a, y=graph.y) for graph in self.data]


def filter_label(dataset, label):
    # divide dataset based on the label[]

    output = []
    res = []

    for graph in dataset:
        if graph.y in label:
            output.append(graph)
        else:
            res.append(graph)

    output_dataset = list_to_spektral_dataset(output)
    res_dataset = list_to_spektral_dataset(res)
    return output_dataset, res_dataset


def train_test_split(dataset, train_percentage):
    # Train/test split
    idxs = np.random.permutation(len(dataset))
    split = int(train_percentage * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

    return dataset_tr, dataset_te


def subsample(dataset, size):
    # Train/test split
    idxs = np.random.permutation(len(dataset))
    split = int(size)
    idx_sample, idx_re = np.split(idxs, [split])
    dataset_sample, dataset_re = dataset[idx_sample], dataset[idx_re]

    return dataset_sample, dataset_re


def merge_dataset(dataset1, dataset2):
    # implemented fucntions in Dataset class of spektral graph
    merged = dataset1.__add__(dataset2)
    return merged


def binary_label(dataset, flag_attack):
    for g in dataset:
        # g.y = np.pad(g.y, (0, 1), 'constant')
        if flag_attack:
            y = np.zeros((2,))
            y[1] = 1
            g.y = y
        else:
            y = np.zeros((2,))
            y[0] = 1
            g.y = y

    return dataset


def convert_label_integer(dataset):
    for g in dataset:
        # g.y = np.pad(g.y, (0, 1), 'constant')
        g.y = np.where(g.y == 1.)[0][0]

    return dataset


def obtain_labels(dataset):
    y = []

    for g in dataset:
        y.append(g.y)

    return np.array(y)


def obtain_feature_matrix(dataset):
    graph_x = []

    for g in dataset:
        graph_x.append(np.sum(g.x, axis=0))
    return np.array(graph_x)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape





