import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Created by: Eitan Kosman, BCAI


class GAT(torch.nn.Module):
    """
    A simple graph model that utilizes attention layers
    """

    def __init__(self, num_features, num_classes, num_hidden=8, heads=8):
        """
        @param num_features: number of node features
        @param num_classes: output dimension of the last layer
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, num_hidden, heads=heads)
        self.conv2 = GATConv(num_hidden, num_classes, heads=1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = x.float(), edge_attr.float()
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
