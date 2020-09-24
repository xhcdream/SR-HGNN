import pickle as pk
from ToolScripts.TimeLogger import log
import torch as t
import scipy.sparse as sp
import numpy as np
import os

DGI_MODEL_DICT = {
    "Douban_0.6_1_1500" : "1589476503Douban_0.6_1_1500_0.0_dgi_oh.pkl",
    "Douban_0.6_2_1500" : "1589517886Douban_0.6_2_1500_0.0_dgi_oh.pkl",
    "Douban_0.6_3_1500" : "1589518382Douban_0.6_3_1500_0.0_dgi_oh.pkl",
    "Douban_0.6_4_1500" : "1589518724Douban_0.6_4_1500_0.0_dgi_oh.pkl",
    "Douban_0.6_5_1500" : "1589518996Douban_0.6_5_1500_0.0_dgi_oh.pkl",

    "Douban_0.8_1_1500" : "1589387383Douban_0.8_1_1500_0.0_dgi_oh.pkl",
    "Douban_0.8_2_1500" : "1589461270Douban_0.8_2_1500_0.0_dgi_oh.pkl",
    "Douban_0.8_3_1500" : "1589461546Douban_0.8_3_1500_0.0_dgi_oh.pkl",
    "Douban_0.8_4_1500" : "1589461815Douban_0.8_4_1500_0.0_dgi_oh.pkl",
    "Douban_0.8_5_1500" : "1589462157Douban_0.8_5_1500_0.0_dgi_oh.pkl",

    "CiaoDVD_0.6_1_250" : "1589340207CiaoDVD_0.6_1_250_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_2_250" : "1589349445CiaoDVD_0.6_2_250_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_3_250" : "1589349459CiaoDVD_0.6_3_250_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_4_250" : "1589349472CiaoDVD_0.6_4_250_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_5_250" : "1589349486CiaoDVD_0.6_5_250_0.0_dgi_oh.pkl",

    "CiaoDVD_0.8_1_500" : "1589256354CiaoDVD_0.8_1_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_2_500" : "1589270842CiaoDVD_0.8_2_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_3_500" : "1589270857CiaoDVD_0.8_3_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_4_500" : "1589270873CiaoDVD_0.8_4_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_5_500" : "1589270888CiaoDVD_0.8_5_500_0.0_dgi_oh.pkl",


    "Epinions_0.8_1_1000" : "1589169329Epinions_0.8_1_1000_0.0_dgi_oh.pkl",
    "Epinions_0.8_2_1000" : "1589169219Epinions_0.8_2_1000_0.0_dgi_oh.pkl",
    "Epinions_0.8_3_1000" : "1589191376Epinions_0.8_3_1000_0.0_dgi_oh.pkl",
    "Epinions_0.8_4_1000" : "1589191764Epinions_0.8_4_1000_0.0_dgi_oh.pkl",
    "Epinions_0.8_5_1000" : "1588686260Epinions_0.8_5_1000_0.0_dgi_oh.pkl",
    'Epinions_0.6_1_1000' : "1589046481Epinions_0.6_1_1000_0.0_dgi_oh.pkl",
    'Epinions_0.6_2_1000' : "1589201101Epinions_0.6_2_1000_0.0_dgi_oh.pkl",
    'Epinions_0.6_3_1000' : "1589201224Epinions_0.6_3_1000_0.0_dgi_oh.pkl",
    'Epinions_0.6_4_1000' : "1589201330Epinions_0.6_4_1000_0.0_dgi_oh.pkl",
    'Epinions_0.6_5_1000' : "1589201502Epinions_0.6_5_1000_0.0_dgi_oh.pkl",

    "Douban_0.6_1_250" : "1589553201Douban_0.6_1_250_0.0_dgi_oh.pkl",
    "Douban_0.8_1_250" : "1589553099Douban_0.8_1_250_0.0_dgi_oh.pkl",
    "Douban_0.6_1_500" : "1589476231Douban_0.6_1_500_0.0_dgi_oh.pkl",
    "Douban_0.8_1_500" : "1589387211Douban_0.8_1_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_1_250" : "1589265576CiaoDVD_0.8_1_250_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_1_500" : "1589340221CiaoDVD_0.6_1_500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_1_1000" : "1589340237CiaoDVD_0.6_1_1000_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_1_1500" : "1589340262CiaoDVD_0.6_1_1500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.6_1_2000" : "1589340299CiaoDVD_0.6_1_2000_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_1_1000" : "1589256368CiaoDVD_0.8_1_1000_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_1_1500" : "1589256390CiaoDVD_0.8_1_1500_0.0_dgi_oh.pkl",
    "CiaoDVD_0.8_1_2000" : "1589256423CiaoDVD_0.8_1_2000_0.0_dgi_oh.pkl",
    "Epinions_0.8_1_250" : "1589547741Epinions_0.8_1_250_0.0_dgi_oh.pkl",
    "Epinions_0.8_1_500" : "1589547172Epinions_0.8_1_500_0.0_dgi_oh.pkl",
    "Epinions_0.8_1_1500" : "1589547201Epinions_0.8_1_1500_0.0_dgi_oh.pkl",
    "Epinions_0.8_1_2000" : "1589547285Epinions_0.8_1_2000_0.0_dgi_oh.pkl",
        
    "Epinions_0.6_1_250" : "1589547956Epinions_0.6_1_250_0.0_dgi_oh.pkl",
    "Epinions_0.6_1_500" : "1589547979Epinions_0.6_1_500_0.0_dgi_oh.pkl",
    "Epinions_0.6_1_1500" : "1589548007Epinions_0.6_1_1500_0.0_dgi_oh.pkl",
    "Epinions_0.6_1_2000" : "1589548082Epinions_0.6_1_2000_0.0_dgi_oh.pkl",
    "Douban_0.6_1_1000" : "1589476343Douban_0.6_1_1000_0.0_dgi_oh.pkl",
    "Douban_0.6_1_2000" : "1589476961Douban_0.6_1_2000_0.0_dgi_oh.pkl",
    "Douban_0.8_1_1000" : "1589387222Douban_0.8_1_1000_0.0_dgi_oh.pkl",
    "Douban_0.8_1_2000" : "1589387713Douban_0.8_1_2000_0.0_dgi_oh.pkl",
}

def mkdir(dataset):
    DIR = os.path.join(os.getcwd(), "Model", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    # DIR = os.path.join(os.getcwd(), "History", dataset)
    # if not os.path.exists(DIR):
    #     os.makedirs(DIR)

def loadData(datasetStr, rate):
    # DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'mats', str(rate) + '_user' + str(cv))
    DIR = os.path.join(os.getcwd(), "data", datasetStr, 'mats')
    TRAIN_FILE = DIR + '/{0}_train.pkl'.format(rate)
    TEST_FILE  = DIR + '/{0}_test.pkl'.format(rate)
    CV_FILE    = DIR + '/{0}_cv.pkl'.format(rate)
    TRUST_FILE = DIR + '/{0}_trust.pkl'.format(rate)
    log(TRAIN_FILE)
    log(TEST_FILE)
    log(CV_FILE)
    log(TRUST_FILE)
    with open(TRAIN_FILE, 'rb') as fs:
        train = pk.load(fs)
    with open(TEST_FILE, 'rb') as fs:
        test = pk.load(fs)
    with open(CV_FILE, 'rb') as fs:
        cv = pk.load(fs)
    with open(TRUST_FILE, 'rb') as fs:
        trust = pk.load(fs)
    return train, test, cv, trust

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



    