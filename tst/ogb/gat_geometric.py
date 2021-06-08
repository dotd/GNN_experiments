import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        hidden_channels = 16
        self.conv1 = GATConv(num_features, 8, heads=8)
        self.conv2 = GATConv(8, num_classes, heads=1)
        # self.conv3 = GATConv(64, 128)
        # self.conv4 = GATConv(128, 256)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = SplineConv(1, 32, dim=1, kernel_size=5)

        # self.lin1 = nn.Linear(256, 128)
        # self.lin2 = nn.Linear(128, 1)

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


class GAT_Reddit(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT_Reddit, self).__init__()
        self.num_layers = 2
        self.conv1 = GATConv(num_features, 8, heads=8)
        self.conv2 = GATConv(64, num_classes, heads=1)
        self.convs = torch.nn.ModuleList([self.conv1, self.conv2])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x.log_softmax(dim=-1)
