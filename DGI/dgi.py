import torch
import torch.nn as nn
import math
from DGI.gcn import GCN
# from gcn import GCN

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, activation)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, feat, shuf_feat, dgi_weight):
        positive = self.encoder(feat, corrupt=False)
        negative = self.encoder(shuf_feat, corrupt=True)
        # summary = torch.sigmoid(positive.mean(dim=0))
        summary = torch.sigmoid(torch.sum(positive*dgi_weight, dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2
