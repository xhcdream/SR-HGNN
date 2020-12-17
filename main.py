# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import dgl
import math
from DGI.dgi import DGI
import argparse
from HGNN import HGNN
from ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from ToolScripts.utils import normalize_adj
from ToolScripts.utils import loadData
from ToolScripts.utils import generate_sp_ont_hot
from ToolScripts.utils import mkdir
from DGI.dgi import DGI
from dgl import DGLGraph
device_gpu = t.device("cuda")
import time
from MyData import MyData
modelUTCStr = str(int(time.time()))[4:]


class Model():
    def getData(self, args):
        trainMat, testMat, validMat, trustMat = loadData(args.dataset, args.rate)
        a = trainMat + testMat + validMat
        assert a.nnz == (trainMat.nnz + testMat.nnz + validMat.nnz)

        adj_DIR = os.path.join(os.getcwd(), "data", dataset, 'mats')
        adj_path = adj_DIR + '/{0}_multi_item_adj.pkl'.format(args.rate)

        with open(adj_path, 'rb') as fs:
            multi_adj = pickle.load(fs)
        
        return trainMat, testMat, validMat, trustMat, multi_adj
    
    def preTrain(self, trust):
        tmpMat = (trust + trust.T)
        userNum = trust.shape[0]
        # userNum, itemNum = train.shape
        adj = (tmpMat != 0)*1
        adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocsr()
        nodeDegree = np.sum(adj, axis=1)
        degreeSum = np.sum(nodeDegree)
        dgi_weight = t.from_numpy((nodeDegree+1e-6)/degreeSum).float().cuda()

        user_feat_sp_tensor = generate_sp_ont_hot(userNum).cuda()
        in_feats = userNum

        # self.social_graph = dgl.graph(adj)

        edge_src, edge_dst = adj.nonzero()
        self.social_graph = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=trust.shape[0],
                              device=device_gpu)


        dgi = DGI(self.social_graph, in_feats, args.dgi_hide_dim, nn.PReLU()).cuda()
        dgi_optimizer = t.optim.Adam(dgi.parameters(), lr=args.dgi_lr, weight_decay=args.dgi_reg)
        cnt_wait = 0
        best = 1e9
        best_t = 0
        for epoch in range(500):
            dgi.train()
            dgi_optimizer.zero_grad()
            idx = np.random.permutation(userNum)
            shuf_feat = sparse_mx_to_torch_sparse_tensor(sp.eye(userNum).tocsr()[idx]).cuda()

            loss = dgi(user_feat_sp_tensor, shuf_feat, dgi_weight)
            loss.backward()
            dgi_optimizer.step()
            log("%.4f"%(loss.item()), save=False, oneline=True)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                DIR = os.path.join(os.getcwd(), "Model", self.args.dataset)

                path = DIR +  r"/dgi_" + modelUTCStr +  "_" + args.dataset + "_" + str(args.rate) + "_" + str(args.dgi_hide_dim) + "_" + str(args.dgi_reg)
                path += '.pth'
                t.save(dgi.state_dict(), path)
                # t.save(dgi, path)
            else:
                cnt_wait += 1

            if cnt_wait == 5:
                print('DGI Early stopping!')
                print(path)
                return path


    def __init__(self, args):
        self.args = args
        train, test, valid, trust, multi_adj = self.getData(self.args)

        tmpMat = (trust + trust.T)
        userNum = trust.shape[0]
        social_adj = (tmpMat != 0)*1
        #add self->self
        social_adj = trust + sp.eye(trust.shape[0])
        social_adj = (social_adj != 0) * 1
        social_adj = social_adj.tocsr()
        edge_src, edge_dst = social_adj.nonzero()
        self.social_graph = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=trust.shape[0],
                              device=device_gpu)


        #pre train social
        self.dgi_path = self.preTrain(trust)
        self.userNum, self.itemNum = train.shape
        self.ratingClass = np.unique(train.data).size
        log("user num =%d, item num =%d"%(self.userNum, self.itemNum))

        self.multi_adj = multi_adj
        item_degree = t.from_numpy((np.sum(multi_adj, axis=1).A != 0) *1)
        self.att_mask = item_degree.view(-1,self.ratingClass).float().to(device_gpu)


        tmpTrust = (trust + trust.T)
        tmpTrust = (tmpTrust != 0)*1

        a = csr_matrix((multi_adj.shape[1], multi_adj.shape[1]))
        b = csr_matrix((multi_adj.shape[0], multi_adj.shape[0]))
        multi_uv_adj = sp.vstack([sp.hstack([a, multi_adj.T]), sp.hstack([multi_adj,b])])

        #train test valid data
        train_coo = train.tocoo()
        test_coo = test.tocoo()
        valid_coo = valid.tocoo()

        self.train_u, self.train_v, self.train_r = train_coo.row, train_coo.col, train_coo.data
        self.test_u, self.test_v, self.test_r = test_coo.row, test_coo.col, test_coo.data
        self.valid_u, self.valid_v, self.valid_r = valid_coo.row, valid_coo.col, valid_coo.data

        self.MyDataLoader = MyData(train, trust, self.args.seed, num_ng=1, is_training=True)

        assert np.sum(self.train_r == 0) == 0
        assert np.sum(self.test_r == 0) == 0
        assert np.sum(self.valid_r == 0) == 0

        self.trainMat = train
        self.testMat  = test
        self.validMat = valid
        self.trustMat = trust

        #normalize 
        self.adj = normalize_adj(multi_uv_adj + sp.eye(multi_uv_adj.shape[0])) 
        self.adj_sp_tensor = sparse_mx_to_torch_sparse_tensor(self.adj).cuda()

        self.att_adj = sparse_mx_to_torch_sparse_tensor(self.trainMat.T != 0).float().cuda()
        self.att_adj_norm = t.from_numpy(np.sum(self.trainMat.T!=0, axis=1).astype(np.float)).float().cuda()

        self.hide_dim = eval(self.args.layer)[0]
        self.r_weight = self.args.r
        self.loss_rmse = nn.MSELoss(reduction='sum')#不求平均
        self.lr = self.args.lr #0.001
        self.decay = self.args.decay
        self.curEpoch = 0
        #history
        self.train_losses = []
        self.train_RMSEs  = []
        self.train_MAEs   = []
        self.test_losses  = []
        self.test_RMSEs   = []
        self.test_MAEs    = []
        self.step_rmse    = []
        self.step_mae     = []
    
    def setRandomSeed(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)


    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        #set random seed
        self.setRandomSeed()

        self.out_dim = sum(eval(self.args.layer))
        self.embed_layer = HGNN(self.userNum, self.itemNum, \
                            self.userNum, self.args.dgi_hide_dim, \
                            self.itemNum*self.ratingClass, self.hide_dim, \
                            layer=self.args.layer, alpha=0.1).cuda()

        self.predLayer = nn.Sequential(
            nn.Linear(self.out_dim*2, self.out_dim*1),
            nn.ReLU(),
            nn.Linear(self.out_dim*1, 1),
            nn.ReLU()
        ).cuda()
        
        self.w_r = nn.Sequential(
            nn.Linear(self.out_dim*2, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, 1, bias=False)
        ).cuda()

        self.w_t = nn.Sequential(
            nn.Linear(self.out_dim*2, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, 1, bias=False)
        ).cuda()

        #one-hot feature
        self.item_feat_sp_tensor = generate_sp_ont_hot(self.itemNum*self.ratingClass).cuda()
        self.user_feat_sp_tensor = generate_sp_ont_hot(self.userNum).cuda()

        self.dgi = DGI(self.social_graph, self.userNum, self.args.dgi_hide_dim, nn.PReLU()).cuda()

        self.dgi.load_state_dict(t.load(self.dgi_path))

        log("load dgi model %s"%(self.dgi_path))
        self.user_dgi_feat = self.dgi.encoder(self.user_feat_sp_tensor).detach()
        if self.args.dgi_norm == 1:
            self.user_dgi_feat = F.normalize(self.user_dgi_feat, p=2, dim=1)
        
        #weight_dict have different reg weight
        weight_dict_params = list(map(id, self.embed_layer.weight_dict.parameters()))
        base_params = filter(lambda p: id(p) not in weight_dict_params, self.embed_layer.parameters())

        self.opt = t.optim.Adam([
            {'params': base_params, 'weight_decay': self.args.r},
            {'params': self.embed_layer.weight_dict.parameters(), 'weight_decay': self.args.r2},
            {'params': self.predLayer.parameters(), 'weight_decay': self.args.r3},
            {'params': self.w_r.parameters(), 'weight_decay': self.args.r},
            {'params': self.w_t.parameters(), 'weight_decay': self.args.r},
        ], lr=self.args.lr)


    def preModel(self, userTensor, itemTensor):
        tensor = t.cat((userTensor, itemTensor), dim=1)
        pred = self.predLayer(tensor)
        return pred
    
    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        validWait = 0
        best_rmse = 9999.0
        best_mae = 9999.0
        rewait_r = 0
        rewait_t = 0
        best_reconstruct_loss_r = 1000000000
        best_reconstruct_loss_t = 1000000000
        for e in range(self.curEpoch, self.args.epochs+1):
            #记录当前epoch,用于保存Model
            self.curEpoch = e
            log("**************************************************************")
            #训练
            epoch_reconstruct_loss_r = 0
            
            epoch_loss, epoch_rmse, epoch_mae, reconstruct_ui_loss, reconstruct_uu_loss  = self.trainModel()
            log("epoch %d/%d, epoch_loss=%.2f, reconstruct_ui_loss=%.4f, reconstruct_uu_loss=%.4f, epoch_rmse=%.4f, epoch_mae=%.4f"% \
                (e,self.args.epochs, epoch_loss, reconstruct_ui_loss, reconstruct_uu_loss, epoch_rmse, epoch_mae))
            
            if reconstruct_ui_loss > 0:
                if reconstruct_ui_loss < best_reconstruct_loss_r:
                    best_reconstruct_loss_r = reconstruct_ui_loss
                    rewait_r = 0
                else:
                    rewait_r += 1
                    log("rewait_r={0}".format(rewait_r))
                
                if rewait_r == self.args.rewait:
                    self.args.lam_r = 0
                    log("stop uv reconstruction")

            if reconstruct_uu_loss > 0:
                if reconstruct_uu_loss < best_reconstruct_loss_t:
                    best_reconstruct_loss_t = reconstruct_uu_loss
                    rewait_t = 0
                else:
                    rewait_t += 1
                    log("rewait_t={0}".format(rewait_t))
                
                if rewait_t == self.args.rewait:
                    self.args.lam_t = 0
                    log("stop uu reconstruction")
            
            self.curLr = self.adjust_learning_rate(self.opt, e+1)

            self.train_losses.append(epoch_loss)
            self.train_RMSEs.append(epoch_rmse)
            self.train_MAEs.append(epoch_mae)
            # valid
            valid_epoch_loss, valid_epoch_rmse, valid_epoch_mae = self.testModel(self.validMat, (self.valid_u, self.valid_v, self.valid_r))
            log("epoch %d/%d, valid_epoch_loss=%.2f, valid_epoch_rmse=%.4f, valid_epoch_mae=%.4f"%(e, self.args.epochs, valid_epoch_loss, valid_epoch_rmse, valid_epoch_mae))
            self.test_losses.append(valid_epoch_loss)
            self.test_RMSEs.append(valid_epoch_rmse)
            self.test_MAEs.append(valid_epoch_mae)
            # test
            test_epoch_loss, test_epoch_rmse, test_epoch_mae = self.testModel(self.testMat, (self.test_u, self.test_v, self.test_r))
            log("epoch %d/%d, test_epoch_loss=%.2f, test_epoch_rmse=%.4f, test_epoch_mae=%.4f"%(e, self.args.epochs, test_epoch_loss, test_epoch_rmse, test_epoch_mae))
            self.step_rmse.append(test_epoch_rmse)
            self.step_mae.append(test_epoch_mae)

            if best_rmse > valid_epoch_rmse:
                best_rmse = valid_epoch_rmse
                best_mae = valid_epoch_mae
                validWait = 0
                best_epoch = self.curEpoch
            else:
                validWait += 1
                log("validWait = %d"%(validWait))

            if self.args.early == 1 and validWait == self.args.patience:
                log('Early stopping! best epoch = %d'%(best_epoch))
                break

    def trainModel(self):
        train_loader = self.MyDataLoader
        log("start negative sample...")
        train_loader.neg_sample()
        log("finish negative sample...")
        userShuffleList = np.random.permutation(self.userNum)
        batch = self.args.batch
        length = self.userNum
        stepCount = math.ceil(length / batch)
        epoch_rmse_loss = 0
        epoch_rmse_num = 0
        epoch_mae_loss = 0
        epoch_reconstruct_ui_loss = 0
        epoch_reconstruct_uu_loss = 0
        for step in range(stepCount):
            beginIdx = step * batch
            endIdx = min((step + 1) * batch, length)
            curStepUserIdx = userShuffleList[beginIdx:endIdx]
            ui_train, uu_train = train_loader.getTrainInstance(curStepUserIdx)

            batch_nodes_u = ui_train[:, 0]
            batch_nodes_v = ui_train[:, 1]
            labels = t.from_numpy(ui_train[:, 2]).float().to(device_gpu)
            neg_label = t.from_numpy(ui_train[:, 3]).float().to(device_gpu)

            user_embed, item_muliti_embed = self.embed_layer(self.user_dgi_feat, self.user_feat_sp_tensor, self.item_feat_sp_tensor, self.adj_sp_tensor)
            item_muliti_embed = item_muliti_embed.view(-1, self.ratingClass, self.out_dim)
            #mean or attention
            item_embed = t.div(t.sum(item_muliti_embed, dim=1), self.ratingClass)
            if self.args.lam_r != 0:
                reconstruct_pos = self.w_r(t.cat((user_embed[batch_nodes_u], item_muliti_embed[batch_nodes_v, ui_train[:, 2]-1]), dim=1))
                reconstruct_neg = self.w_r(t.cat((user_embed[batch_nodes_u], item_muliti_embed[batch_nodes_v, ui_train[:, 3]-1]), dim=1))
                reconstruct_loss = (- (reconstruct_pos.view(-1) - reconstruct_neg.view(-1)).sigmoid().log().sum())
                epoch_reconstruct_ui_loss += reconstruct_loss.item()

            if self.args.lam_t != 0:
                trust_uid = uu_train[:, 0]
                trust_tid = uu_train[:, 1]
                trust_neg_uid = uu_train[:, 2]
                reconstruct_pos_t = self.w_t(t.cat((user_embed[trust_uid], user_embed[trust_tid]), dim=1))
                reconstruct_neg_t = self.w_t(t.cat((user_embed[trust_uid], user_embed[trust_neg_uid]), dim=1))
                trust_reconstruct_loss = (- (reconstruct_pos_t.view(-1) - reconstruct_neg_t.view(-1)).sigmoid().log().sum())
                epoch_reconstruct_uu_loss += trust_reconstruct_loss.item()


            userEmbed = user_embed[batch_nodes_u]
            itemEmbed = item_embed[batch_nodes_v]

            pred = self.preModel(userEmbed, itemEmbed)

            loss = self.loss_rmse(pred.view(-1), labels)

            epoch_rmse_loss += loss.item()
            epoch_mae_loss += t.sum(t.abs(pred.view(-1) - labels)).item()
            epoch_rmse_num += batch_nodes_u.size
            
            curBathch = ui_train.shape[0]
            loss = loss/curBathch

            if self.args.lam_r != 0:
                loss += ((reconstruct_loss*self.args.lam_r)/curBathch)
            
            if self.args.lam_t != 0:
                loss += ((trust_reconstruct_loss*self.args.lam_t)/uu_train.shape[0])

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('setp %d/%d, step_loss = %f'%(step, stepCount, loss.item()), save=False, oneline=True)
        epoch_rmse = np.sqrt(epoch_rmse_loss / epoch_rmse_num)
        epoch_mae = epoch_mae_loss / epoch_rmse_num

        epoch_reconstruct_ui_loss = epoch_reconstruct_ui_loss/stepCount
        epoch_reconstruct_uu_loss = epoch_reconstruct_uu_loss/stepCount
        return epoch_rmse_loss, epoch_rmse, epoch_mae, epoch_reconstruct_ui_loss, epoch_reconstruct_uu_loss


    def testModel(self, testMat, data):
        test_u, test_v, test_r = data
        batch = self.args.batch
        num = len(test_u)
        assert testMat.nnz == num
        # shuffledIds = np.random.permutation(num)
        shuffledIds = np.arange(num)
        steps = int(np.ceil(num / batch))
        epoch_rmse_loss = 0
        epoch_rmse_num = 0
        epoch_mae_loss = 0
        with t.no_grad():
            user_embed, item_muliti_embed = self.embed_layer(self.user_dgi_feat, self.user_feat_sp_tensor, self.item_feat_sp_tensor, self.adj_sp_tensor)
            item_muliti_embed = item_muliti_embed.view(-1, self.ratingClass, self.out_dim)

            item_embed = t.div(t.sum(item_muliti_embed, dim=1), self.ratingClass)
        
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            batch_nodes_u = test_u[batch_ids]
            batch_nodes_v = test_v[batch_ids]
            labels_list = t.from_numpy(test_r[batch_ids]).float().to(device_gpu)

            userEmbedSteps = user_embed[batch_nodes_u]
            itemEmbedSteps = item_embed[batch_nodes_v]
            with t.no_grad():
                pred = self.preModel(userEmbedSteps, itemEmbedSteps)
                loss = self.loss_rmse(pred.view(-1), labels_list)

            epoch_rmse_loss += loss.item()
            epoch_mae_loss += t.sum(t.abs(pred.view(-1) - labels_list)).item()
            epoch_rmse_num += batch_nodes_u.size

        epoch_rmse = np.sqrt(epoch_rmse_loss / epoch_rmse_num)
        epoch_mae = epoch_mae_loss / epoch_rmse_num
        return epoch_rmse_loss, epoch_rmse, epoch_mae



    #根据epoch数调整学习率
    def adjust_learning_rate(self, opt, epoch):
        if opt != None:
            for param_group in opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, 0.0001)
        return 1

    def getModelName(self):
        title = "SR-HGNN_"
        ModelName = title + dataset + "_" + modelUTCStr + \
        "_rate" + str(self.args.rate) + \
        "_reg_" + str(self.args.r)+ \
        "_gcn_r_" + str(self.args.r2)+ \
        "_pred_r_" + str(self.args.r3)+ \
        "_batch_" + str(self.args.batch) + \
        "_lamr_" + str(self.args.lam_r) +\
        "_lamt_" + str(self.args.lam_t) +\
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_ufeat_" + str(self.args.dgi_hide_dim) +\
        "_Layer_" + self.args.layer
        return ModelName



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-HGNN main.py')
    #dataset params
    parser.add_argument('--dataset', type=str, default="CiaoDVD", help="CiaoDVD,Epinions,Douban")
    parser.add_argument('--rate', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--r', type=float, default=0.001)
    parser.add_argument('--r2', type=float, default=0.2)
    parser.add_argument('--r3', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--epochs', type=int, default=200)
    #early stop params
    parser.add_argument('--early', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--rewait', type=int, default=5)
    
    #reconstruction params
    parser.add_argument('--lam_r', type=float, default=0.1)
    parser.add_argument('--lam_t', type=float, default=0)

    parser.add_argument('--layer', type=str, default="[16,16]")

    #dgi params
    parser.add_argument('--dgi_norm', type=int, default=0)
    parser.add_argument('--dgi_hide_dim', type=int, default=500)
    parser.add_argument('--dgi_lr', type=float, default=0.001)
    parser.add_argument('--dgi_reg', type=float, default=0)
    
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    mkdir(dataset)

    hope = Model(args)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()

