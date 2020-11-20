import pickle 
import numpy as np
import scipy.sparse as sp
import random
import os
import argparse
from create_adj import creatMultiItemUserAdj

#python splitTrainTestvalid --dataset CiaoDVD --rate 0.8
parser = argparse.ArgumentParser()
#dataset params
parser.add_argument('--dataset', type=str, default="Epinions", help="CiaoDVD,Epinions,Douban")
parser.add_argument('--rate', type=float, default=0.8, help="0.8 or 0.6")
args = parser.parse_args()

dataset = args.dataset
rate = args.rate

DIR = os.path.join(os.getcwd(), "data", dataset)


with open(DIR + "/ratings.pkl", 'rb') as fs:
    data = pickle.load(fs)

row, col = data.shape
print("user num = %d, item num = %d"%(row, col))

uid_list = data.tocoo().row
iid_list = data.tocoo().col
rating_list = data.tocoo().data
l = np.random.permutation(uid_list.size)
size_train = int(uid_list.size * rate)
size_test = int(uid_list.size * (1-rate)/2)


train_idx = l[0: size_train]
test_idx = l[size_train: size_train + size_test]
valid_idx = l[size_train + size_test:]
sp.coo_matrix

train = sp.csc_matrix((rating_list[train_idx], (uid_list[train_idx], iid_list[train_idx])), shape=data.shape)

test = sp.csc_matrix((rating_list[test_idx], (uid_list[test_idx], iid_list[test_idx])), shape=data.shape)

valid = sp.csc_matrix((rating_list[valid_idx], (uid_list[valid_idx], iid_list[valid_idx])), shape=data.shape)

assert data.nnz == (train + test + valid).nnz


print("train num = %d"%(train.nnz))
print("test num = %d"%(test.nnz))
print("valid num = %d"%(valid.nnz))

with open(DIR + "/mats/{0}_train.pkl".format(rate), 'wb') as fs:
    pickle.dump(train.tocsr(), fs)
with open(DIR + "/mats/{0}_test.pkl".format(rate), 'wb') as fs:
    pickle.dump(test.tocsr(), fs)
with open(DIR + "/mats/{0}_valid.pkl".format(rate), 'wb') as fs:
    pickle.dump(valid.tocsr(), fs)

#filter
with open(DIR + "/mats/{0}_train.pkl".format(rate), 'rb') as fs:
    train = pickle.load(fs)
with open(DIR + "/mats/{0}_test.pkl".format(rate), 'rb') as fs:
    test = pickle.load(fs)
with open(DIR + "/mats/{0}_valid.pkl".format(rate), 'rb') as fs:
    valid = pickle.load(fs)
with open(DIR + "/trust.pkl", 'rb') as fs:
    trust = pickle.load(fs)

a = np.sum(np.sum(train != 0, axis=1) ==0)
b = np.sum(np.sum(train != 0, axis=0) ==0)
c = np.sum(np.sum(trust, axis=1) == 0)
while a != 0 or b != 0 or c != 0:
    if a != 0:
        idx, _ = np.where(np.sum(train != 0, axis=1) != 0)
        train = train[idx]
        test = test[idx]
        valid = valid[idx]
        trust = trust[idx][:, idx]
    elif b != 0:
        _, idx = np.where(np.sum(train != 0, axis=0) != 0)
        train = train[:, idx]
        test = test[:, idx]
        valid = valid[:, idx]
    elif c != 0:
        idx, _ = np.where(np.sum(trust, axis=1) != 0)
        train = train[idx]
        test = test[idx]
        valid = valid[idx]
        trust = trust[idx][:, idx]
    a = np.sum(np.sum(train != 0, axis=1) ==0)
    b = np.sum(np.sum(train != 0, axis=0) ==0)
    c = np.sum(np.sum(trust, axis=1) == 0)


with open(DIR + "/mats/{0}_train.pkl".format(rate), 'wb') as fs:
    pickle.dump(train, fs)
with open(DIR + "/mats/{0}_test.pkl".format(rate), 'wb') as fs:
    pickle.dump(test, fs)
with open(DIR + "/mats/{0}_valid.pkl".format(rate), 'wb') as fs:
    pickle.dump(valid, fs)
with open(DIR + "/mats/{0}_trust.pkl".format(rate), 'wb') as fs:
    pickle.dump(trust, fs)

creatMultiItemUserAdj(args.dataset, args.rate)
print("Done")