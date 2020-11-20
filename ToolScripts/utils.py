import pickle as pk
from ToolScripts.TimeLogger import log
import torch as t
import scipy.sparse as sp
import numpy as np
import os


def mkdir(dataset):
    DIR = os.path.join(os.getcwd(), "Model", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    # DIR = os.path.join(os.getcwd(), "History", dataset)
    # if not os.path.exists(DIR):
    #     os.makedirs(DIR)

def loadData(datasetStr, rate):
    DIR = os.path.join(os.getcwd(), "data", datasetStr, 'mats')
    TRAIN_FILE = DIR + '/{0}_train.pkl'.format(rate)
    TEST_FILE  = DIR + '/{0}_test.pkl'.format(rate)
    VALID_FILE    = DIR + '/{0}_valid.pkl'.format(rate)
    TRUST_FILE = DIR + '/{0}_trust.pkl'.format(rate)
    log(TRAIN_FILE)
    log(TEST_FILE)
    log(VALID_FILE)
    log(TRUST_FILE)
    with open(TRAIN_FILE, 'rb') as fs:
        train = pk.load(fs)
    with open(TEST_FILE, 'rb') as fs:
        test = pk.load(fs)
    with open(VALID_FILE, 'rb') as fs:
        valid = pk.load(fs)
    with open(TRUST_FILE, 'rb') as fs:
        trust = pk.load(fs)
    return train, test, valid, trust

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def generate_sp_ont_hot(num):
    mat = sp.eye(num)
    # mat = sp.dok_matrix((num, num))
    # for i in range(num):
    #     mat[i,i] = 1
    ret = sparse_mx_to_torch_sparse_tensor(mat)
    return ret



    