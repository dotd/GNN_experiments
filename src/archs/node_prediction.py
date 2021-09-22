import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, ARMAConv
import torch.nn.functional as F


class NodeGat(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden, num_heads, n_hidden_layers=2, dropout=0.5):

        super(NodeGat, self).__init__()
        self.conv1 = GATConv(num_features, num_hidden, num_heads)
        self.conv2 = nn.ModuleList([GATConv(num_hidden * num_heads, num_hidden, num_heads) for _ in range(n_hidden_layers - 1)])
        self.clf = nn.Linear(num_hidden * num_heads, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, training=self.training, p=self.dropout)
        for l in self.conv2:
            x = l(x, edge_index)
            x = F.leaky_relu(x, 0.2)

        x = self.clf(x)
        return F.log_softmax(x, dim=1)


class NodeGCN(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden, num_layers, apply_log_softmax=True):
        super(NodeGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.apply_log_softmax = apply_log_softmax
        for i in range(num_layers):
            in_features = num_features if i == 0 else num_hidden
            out_features = num_classes if i == num_layers - 1 else num_hidden
            self.convs.append(GCNConv(in_features, out_features))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x)

        if self.apply_log_softmax:
            return F.log_softmax(x, dim=1)

        return x


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
