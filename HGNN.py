import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class HGNN(nn.Module):
    def __init__(self, userNum, itemNum, \
                user_feat, user_social_feat, item_in_feat, hide_dim,\
                layer="[16,16,16]", alpha=0.1, dgi=True):
        super(HGNN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        initializer = nn.init.xavier_uniform_
        
        self.layer = eval(layer)
        self.layerNum = len(self.layer)

        self.user1_w = nn.Linear(user_feat, int(self.hide_dim/2), bias=False)
        self.user2_w = nn.Linear(user_social_feat, int(self.hide_dim/2), bias=False)
        initializer(self.user1_w.weight)
        initializer(self.user2_w.weight)
            
        self.item_w = nn.Linear(item_in_feat, hide_dim, bias=False)
        initializer(self.item_w.weight)

        self.weight_dict = nn.ParameterDict()
        for k in range(1, self.layerNum):
            if k == 0:
                self.weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(t.empty(self.hide_dim, self.layer[k])))})
                self.weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(t.empty(self.hide_dim, self.layer[k])))})
            else:
                self.weight_dict.update({'user_w%d'%k: nn.Parameter(initializer(t.empty(self.layer[k-1], self.layer[k])))})
                self.weight_dict.update({'item_w%d'%k: nn.Parameter(initializer(t.empty(self.layer[k-1], self.layer[k])))})

        # self.act = t.nn.LeakyReLU(alpha)
        self.act = t.nn.PReLU()

    def forward(self, user_social_feat, user_feat, item_feat, raitng_adj):
        item_e = self.item_w(item_feat)
        user_e1 = self.user1_w(user_feat)
        user_e2 = self.user2_w(user_social_feat)
        user_e = t.cat((user_e1, user_e2), dim=1)

        ego_embeddings = t.cat((user_e, item_e), dim=0)
        embeddings = self.act(t.spmm(raitng_adj, ego_embeddings))
        
        # orignal embedding
        all_user_embeddings = [embeddings[: self.userNum]]
        all_item_embeddings = [embeddings[self.userNum: ]]

        for k in range(1, self.layerNum):
            tmp_user_embed = t.mm(all_user_embeddings[-1], self.weight_dict['user_w%d'%k])
            tmp_item_embed = t.mm(all_item_embeddings[-1], self.weight_dict['item_w%d'%k])

            ego_embeddings = t.cat((tmp_user_embed, tmp_item_embed), dim=0)
            embeddings = self.act(t.spmm(raitng_adj, ego_embeddings))

            all_user_embeddings += [embeddings[: self.userNum]]
            all_item_embeddings += [embeddings[self.userNum: ]]

        
        user_embedding = t.cat(all_user_embeddings, 1)
        item_embedding = t.cat(all_item_embeddings, 1)
        
        return user_embedding, item_embedding




        

