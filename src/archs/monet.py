import math

import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool, GMMConv)
import torch_geometric.transforms as T

"""
From: https://github.com/theswgong/MoNet/blob/master/conv/gmm_conv.py
"""

EPS = 1e-15


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


# class GMMConv(MessagePassing):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  dim,
#                  kernel_size,
#                  bias=True,
#                  **kwargs):
#         super(GMMConv, self).__init__(aggr='add', **kwargs)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dim = dim
#         self.kernel_size = kernel_size
#
#         self.lin = torch.nn.Linear(in_channels,
#                                    out_channels * kernel_size,
#                                    bias=False)
#         self.mu = Parameter(torch.Tensor(kernel_size, dim))
#         self.sigma = Parameter(torch.Tensor(kernel_size, dim))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.mu)
#         glorot(self.sigma)
#         zeros(self.bias)
#         reset(self.lin)
#
#     def forward(self, x, edge_index, pseudo):
#         x = x.unsqueeze(-1) if x.dim() == 1 else x
#         pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
#
#         out = self.lin(x).view(-1, self.kernel_size, self.out_channels)
#         out = self.propagate(edge_index, x=out, pseudo=pseudo)
#
#         if self.bias is not None:
#             out = out + self.bias
#         return out
#
#     def message(self, x_j, pseudo):
#         (E, D), K = pseudo.size(), self.mu.size(0)
#
#         gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D)) ** 2
#         gaussian = gaussian / (EPS + self.sigma.view(1, K, D) ** 2)
#         gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]
#
#         return (x_j * gaussian).sum(dim=1)
#
#     def __repr__(self):
#         return '{}({}, {}, kernel_size={})'.format(self.__class__.__name__,
#                                                    self.in_channels,
#                                                    self.out_channels,
#                                                    self.kernel_size)


class MoNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        data = data.clone()
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)
