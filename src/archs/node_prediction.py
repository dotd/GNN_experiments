import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, ARMAConv
import torch.nn.functional as F


class NodeGat(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden, num_heads):
        super(NodeGat, self).__init__()
        self.conv1 = GATConv(num_features, num_hidden, num_heads)
        self.conv2 = GATConv(num_hidden * num_heads, num_hidden, num_heads)
        self.clf = nn.Linear(num_hidden * num_heads, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.clf(x)
        return F.log_softmax(x, dim=1)


class NodeGCN(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden):
        super(NodeGCN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class NodeARMA(nn.Module):
    def __init__(self, num_features, num_classes):
        super(NodeARMA, self).__init__()

        self.conv1 = ARMAConv(num_features, 16, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(16, num_classes, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
