import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from .conv import GNN_node


class GCN(torch.nn.Module):

    def __init__(self, num_classes: int, num_layer=5, emb_dim=300, node_encoder=None,
                 edge_encoder_ctor: torch.nn.Module = None,
                 residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean", max_seq_len=1):

        super(GCN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, emb_dim=emb_dim, JK=JK, node_encoder=node_encoder,
                                 edge_encoder_ctor=edge_encoder_ctor, drop_ratio=drop_ratio,
                                 residual=residual)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()
        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_classes))

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))
        preds = pred_list[0] if self.max_seq_len == 1 else pred_list
        return preds
