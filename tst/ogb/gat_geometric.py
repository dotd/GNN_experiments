import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        hidden_channels = 16
        self.conv1 = GATConv(num_features, 32)
        self.conv2 = GATConv(32, 64)
        self.conv3 = GATConv(64, 128)
        self.conv4 = GATConv(128, 256)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = SplineConv(1, 32, dim=1, kernel_size=5)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 1)

        # self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        # self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False,
        #                      dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = x.float(), edge_attr.float()
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.conv4(x, edge_index)
        x = F.elu(x)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        return x
