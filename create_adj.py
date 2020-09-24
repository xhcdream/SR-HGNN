import numpy as np
import pickle 
import scipy.sparse as sp
import os

from ToolScripts.utils import loadData

def creatMultiItemUserAdj(dataset, rate):
    trainMat, _, _, _ = loadData(dataset, rate)
    ratingClass = np.unique(trainMat.data).size
    userNum, itemNum = trainMat.shape
    mult_adj = sp.lil_matrix((ratingClass*itemNum, userNum), dtype=np.int)
    uidList = trainMat.tocoo().row
    iidList = trainMat.tocoo().col
    rList = trainMat.tocoo().data

    for i in range(uidList.size):
        uid = uidList[i]
        iid = iidList[i]
        r = rList[i]
        mult_adj[iid*ratingClass+r-1, uid] = 1

    DIR = os.path.join(os.getcwd(), "data", dataset, 'mats')
    path = DIR + '/{0}_multi_item_adj.pkl'.format(rate)
    with open(path, 'wb') as fs:
        pickle.dump(mult_adj.tocsr(), fs)
    print("create multi_item_feat")