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
from dgl import DGLGraph
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
    
    def preTrain(self, train, trust):
        tmpMat = (trust + trust.T)
        userNum, itemNum = train.shape
        adj = (tmpMat != 0)*1
        adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocsr()
        nodeDegree = np.sum(adj, axis=1)
        degreeSum = np.sum(nodeDegree)
        dgi_weight = t.from_numpy((nodeDegree+1e-6)/degreeSum).float().cuda()

        user_feat_sp_tensor = generate_sp_ont_hot(userNum).cuda()
        in_feats = userNum

        self.social_graph = DGLGraph(adj)
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
        #pre train dgi model
        self.dgi_path = self.preTrain(train, trust)


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

    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)

        self.out_dim = sum(eval(self.args.layer))
        self.embed_layer = HGNN(self.userNum, self.itemNum, \
                            self.userNum, self.args.dgi_hide_dim, \
                            self.itemNum*self.ratingClass, self.hide_dim, \
                            layer=self.args.layer, alpha=0.1).cuda()
        

        self.predLayer = nn.Sequential(
            nn.Linear(self.out_dim*2, self.out_dim*1),
            nn.ReLU(),
            nn.Linear(self.out_dim*1, 1)
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
            epoch_loss, epoch_rmse, epoch_mae, epoch_reconstruct_loss_r  = self.trainModel((self.train_u, self.train_v, self.train_r))
            log("epoch %d/%d, epoch_loss=%.2f, epoch_reconstruct_loss=%.4f, epoch_rmse=%.4f, epoch_mae=%.4f"% \
                (e,self.args.epochs, epoch_loss, epoch_reconstruct_loss_r, epoch_rmse, epoch_mae))
            
            if epoch_reconstruct_loss_r > 0:
                if epoch_reconstruct_loss_r < best_reconstruct_loss_r:
                    best_reconstruct_loss_r = epoch_reconstruct_loss_r
                    rewait_r = 0
                else:
                    rewait_r += 1
                    log("rewait{0}".format(rewait_r))
                
                if rewait_r == self.args.rewait:
                    self.args.lam_r = 0
                    log("stop uv reconstruction")
            
            epoch_reconstruct_loss_t = 0
            if self.args.lam_t != 0:
                epoch_reconstruct_loss_t = self.trainSocial(self.trustMat)
                log("epoch %d/%d, epoch_reconstruct_social_loss=%.2f"% (e,self.args.epochs, epoch_reconstruct_loss_t))
                if epoch_reconstruct_loss_t < best_reconstruct_loss_t:
                    best_reconstruct_loss_t = epoch_reconstruct_loss_t
                    rewait_t = 0
                else:
                    rewait_t += 1
                    log("rewait_t{0}".format(rewait_t))
                
                if rewait_t == self.args.rewait:
                    self.args.lam_t = 0
                    log("stop uu reconstruction")
            
            self.opt
            self.curLr = self.adjust_learning_rate(self.opt, e+1)

            self.train_losses.append(epoch_loss)
            self.train_RMSEs.append(epoch_rmse)
            self.train_MAEs.append(epoch_mae)
            valid_epoch_loss, valid_epoch_rmse, valid_epoch_mae = self.testModel(self.validMat, (self.valid_u, self.valid_v, self.valid_r))
            log("epoch %d/%d, valid_epoch_loss=%.2f, valid_epoch_rmse=%.4f, valid_epoch_mae=%.4f"%(e, self.args.epochs, valid_epoch_loss, valid_epoch_rmse, valid_epoch_mae))
            #验证
            self.test_losses.append(valid_epoch_loss)
            self.test_RMSEs.append(valid_epoch_rmse)
            self.test_MAEs.append(valid_epoch_mae)
            #测试
            all_epoch_loss, all_epoch_rmse, all_epoch_mae = self.testModel(self.testMat, (self.test_u, self.test_v, self.test_r))
            log("epoch %d/%d, all_epoch_loss=%.2f, all_epoch_rmse=%.4f, all_epoch_mae=%.4f"%(e, self.args.epochs, all_epoch_loss, all_epoch_rmse, all_epoch_mae))
            self.step_rmse.append(all_epoch_rmse)
            self.step_mae.append(all_epoch_mae)

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


    # def ng_sample2(self, train_r, ng_num=1):
    #     ng_r = []
    #     # tmp = np.array([1,2,3,4,5])
    #     tmp = np.arange(1, self.ratingClass+1)
    #     for r in train_r:
    #         arr = np.delete(tmp, int(r-1))
    #         neg = np.random.choice(arr, ng_num, replace=False)
    #         ng_r.append(neg)
    #     return np.array(ng_r)

    def ng_sample(self, train_r, ng_num=1):
        ng_r = []
        num = train_r.size
        tmp = train_r.astype(np.int)
        res = np.random.randint(1, self.ratingClass+1, num)

        rebuild_idx = np.where(tmp == res)[0]
        for idx in rebuild_idx:
            val = np.random.randint(1, self.ratingClass+1)
            while val == tmp[idx]:
                val = np.random.randint(1, self.ratingClass+1)
            res[idx] = val
        assert np.sum(res == tmp) == 0
        return res

    def ng_social_sample2(self, uid, tid, trust, ng_num=1):
        tmp = trust.todok()
        ret_ng_sample = []
        for i in uid:
            l = []
            for t in range(ng_num):
                j = np.random.randint(self.userNum)
                while (i, j) in tmp:
                    j = np.random.randint(self.userNum)
                l.append(j)
            ret_ng_sample.append(l)
        return np.array(ret_ng_sample)

    def ng_social_sample(self, uid, tid, trust):
        tmpTrustMat = trust.todok()
        num = uid.size
        userNum = trust.shape[0]
        neg = np.random.randint(low=0, high=userNum, size=num)
        for i in range(num):
            user_id = uid[i]
            item_id = neg[i]
            if (user_id, item_id) in tmpTrustMat:
                while (user_id, item_id) in tmpTrustMat:
                    item_id = np.random.randint(low=0, high=userNum)
                neg[i] = item_id
            else:
                continue
        return neg


    def trainSocial(self, trust):
        train_uid = trust.tocoo().row
        train_tid = trust.tocoo().col
        ng = self.ng_social_sample(train_uid, train_tid, trust)
        # ng2 = self.ng_social_sample2(train_uid, train_tid, trust)
        batch = self.args.batch
        num = len(train_uid)
        shuffledIds = np.random.permutation(num)
        steps = int(np.ceil(num / batch))
        epoch_loss = 0
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            batch_nodes_u = train_uid[batch_ids]
            batch_nodes_v = train_tid[batch_ids]
            user_embed, _ = self.embed_layer(self.user_dgi_feat, self.user_feat_sp_tensor, self.item_feat_sp_tensor, self.adj_sp_tensor)
            reconstruct_pos = self.w_t(t.cat((user_embed[batch_nodes_u], user_embed[batch_nodes_v]), dim=1))
            reconstruct_neg = self.w_t(t.cat((user_embed[batch_nodes_u], user_embed[ng[batch_ids]]), dim=1))
            reconstruct_loss = (- (reconstruct_pos.view(-1) - reconstruct_neg.view(-1)).sigmoid().log().sum())

            loss = reconstruct_loss * self.args.lam_t / batch
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            log('setp %d/%d, step_loss = %f'%(i, steps, loss.item()), save=False, oneline=True)
        return epoch_loss
        

    
    def trainModel(self, data):
        train_u, train_v, train_r = data
        train_r = train_r.astype(np.int)
        if self.args.lam_r != 0:
            ng_r = self.ng_sample(train_r)
        batch = self.args.batch
        num = len(train_u)
        assert self.trainMat.nnz == num
        shuffledIds = np.random.permutation(num)
        steps = int(np.ceil(num / batch))
        epoch_rmse_loss = 0
        epoch_rmse_num = 0
        epoch_mae_loss = 0
        epoch_reconstruct_loss = 0
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            batch_nodes_u = train_u[batch_ids]
            batch_nodes_v = train_v[batch_ids]
            labels_list = t.from_numpy(train_r[batch_ids]).float().to(device_gpu)

            user_embed, item_muliti_embed = self.embed_layer(self.user_dgi_feat, self.user_feat_sp_tensor, self.item_feat_sp_tensor, self.adj_sp_tensor)
            item_muliti_embed = item_muliti_embed.view(-1, self.ratingClass, self.out_dim)
            #mean or attention
            item_embed = t.div(t.sum(item_muliti_embed, dim=1), self.ratingClass)

            if self.args.lam_r != 0:
                reconstruct_pos = self.w_r(t.cat((user_embed[train_u[batch_ids]], item_muliti_embed[train_v[batch_ids], train_r[batch_ids]-1]), dim=1))
                reconstruct_neg = self.w_r(t.cat((user_embed[train_u[batch_ids]], item_muliti_embed[train_v[batch_ids], ng_r[batch_ids]-1]), dim=1))
                reconstruct_loss = (- (reconstruct_pos.view(-1) - reconstruct_neg.view(-1)).sigmoid().log().sum())
                epoch_reconstruct_loss += reconstruct_loss.item()
            userEmbed = user_embed[batch_nodes_u]
            itemEmbed = item_embed[batch_nodes_v]

            pred = self.preModel(userEmbed, itemEmbed)

            loss = self.loss_rmse(pred.view(-1), labels_list)

            epoch_rmse_loss += loss.item()
            epoch_mae_loss += t.sum(t.abs(pred.view(-1) - labels_list)).item()
            epoch_rmse_num += batch_nodes_u.size
            if self.args.lam_r != 0 :
                loss = (loss + reconstruct_loss*self.args.lam_r)/self.args.batch
            else:
                loss = loss/self.args.batch

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('setp %d/%d, step_loss = %f'%(i,steps, loss.item()), save=False, oneline=True)
        epoch_rmse = np.sqrt(epoch_rmse_loss / epoch_rmse_num)
        epoch_mae = epoch_mae_loss / epoch_rmse_num
        epoch_reconstruct_loss = epoch_reconstruct_loss/steps
        return epoch_rmse_loss, epoch_rmse, epoch_mae, epoch_reconstruct_loss

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
    def adjust_learning_rate(self, optimizer, epoch):
        if optimizer != None:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * self.args.decay
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
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--epochs', type=int, default=200)
    #early stop params
    parser.add_argument('--early', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--rewait', type=int, default=5)
    
    #reconstruction params
    parser.add_argument('--lam_r', type=float, default=0)
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

