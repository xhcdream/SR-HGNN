"""Torch modules for graph convolutions(GCN)."""
import torch as th
from torch import nn
from torch.nn import init

import dgl.function as fn
from dgl.base import DGLError

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        graph = graph.local_var()

        degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp)

        weight = self.weight

        feat = th.matmul(feat, self.weight)
        feat = feat * norm
        graph.srcdata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h']
        rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
