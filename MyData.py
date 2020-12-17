import numpy as np
import scipy.sparse as sp
import pickle
import random
from collections import defaultdict

class MyData():
    def __init__(self, trainMat, trustMat, seed, num_ng=0, is_training=None):
        super(MyData, self).__init__()
        self.setRandomSeed(seed)
        self.trainMat  = trainMat
        self.trustMat = trustMat
        self.userNum, self.itemNum = trainMat.shape
        self.num_ng = num_ng
        self.is_training = is_training

        train_u, train_v = self.trainMat.nonzero()
        train_r = self.trainMat.data
        self.ratingClass = np.unique(train_r).size

        assert np.sum(self.trainMat.data == 0) == 0
        self.train_data = np.hstack(
            (train_u.reshape(-1, 1), train_v.reshape(-1, 1), train_r.reshape(-1, 1)))
        self.train_data = self.train_data.astype(np.int)

        train_u1, train_u2 = self.trustMat.nonzero()
        self.trust_data = np.hstack(
            (train_u1.reshape(-1, 1), train_u2.reshape(-1, 1)))
        self.trust_data = self.trust_data.astype(np.int)
    
    def setRandomSeed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def neg_sample(self):
        self.train_neg_sample()
        self.trust_neg_sample()

    def trust_neg_sample(self):
        assert self.is_training
        tmp_trustMat = self.trustMat.todok()
        length = self.trust_data.shape[0]
        trust_neg_data = np.random.randint(low=0, high=self.userNum, size=length)
        self.trust_data_dict = defaultdict(list)

        for i in range(length):
            uid = self.trust_data[i][0]
            neg_fid = trust_neg_data[i]
            if (uid, neg_fid) in tmp_trustMat:
                while (uid, neg_fid) in tmp_trustMat:
                    neg_fid = np.random.randint(low=0, high=self.userNum)
                trust_neg_data[i] = neg_fid

            self.trust_data_dict[uid].append([uid, self.trust_data[i][1], trust_neg_data[i]])

    def train_neg_sample(self):
        #'no need to sampling when testing'
        assert self.is_training
        self.train_data_dict = defaultdict(list)
        
        length = self.trainMat.data.size
        train_data = self.trainMat.data
        train_neg_data = np.random.randint(low=1, high=self.ratingClass+1, size=length)

        # rebuild_idx = np.where(train_data == train_neg_data)[0]
        rebuild_idx = np.where(np.abs(train_data-train_neg_data)<2)[0]
        
        for idx in rebuild_idx:
            val = np.random.randint(1, self.ratingClass+1)
            while val == train_data[idx]:
                val = np.random.randint(1, self.ratingClass+1)
            train_neg_data[idx] = val
        
        assert np.sum(train_data == train_neg_data) == 0
        
        for i in range(length):
            uid = self.train_data[i][0]
            iid = self.train_data[i][1]
            rating = self.train_data[i][2]
            neg_rating = train_neg_data[i]
            self.train_data_dict[uid].append([uid, iid, rating, neg_rating])

    
    def getTrainInstance(self, userIdxList):
        ui_train = []
        uu_train = []

        for uidx in userIdxList:
            ui_train += self.train_data_dict[uidx]
            uu_train += self.trust_data_dict[uidx]
        
        ui_train = np.array(ui_train)
        idx = np.random.permutation(len(ui_train))
        ui_train = ui_train[idx]

        uu_train = np.array(uu_train)
        idx = np.random.permutation(len(uu_train))
        uu_train = uu_train[idx]

        return ui_train, uu_train
