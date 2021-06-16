from collections import OrderedDict

import numpy as np
from torch import nn


# Created by: Eitan Kosman, BCAI
# Implementation of a multi-layer fully connected network with linearly decreasing layer sizes and relu activations


class MLP(nn.Module):
    """
    A module which consists of several n_steps fully connected layers with ReLU activations
    between every 2 layers. The output of the module has no activation
    """

    def __init__(self, in_features: int, out_features: int, n_steps: int = 2, dropout: float = None, bias: bool = True):
        """
        @param in_features - number of input features for the first layer
        @param out_features - number of outputs of the last layer
        @param n_steps - number of layers to construct
        @param n_steps - dropout parameter to use between each two consecutive layers, if provided.
                        If not provided, no dropout would be used.
        @param bias - whether to use bias in each linear layer
        """
        super(MLP, self).__init__()
        steps = np.linspace(in_features, out_features, n_steps + 1, dtype=int)
        steps[0] = in_features
        steps[-1] = out_features
        layers = OrderedDict()
        for idx, (in_f, out_f) in enumerate(zip(steps[: -1], steps[1:])):
            layers[f"linear_{idx}"] = nn.Linear(in_features=in_f, out_features=out_f, bias=bias)
            layers[f"relu_{idx}"] = nn.ReLU()
            if dropout is not None:
                layers[f"dropout_{idx}"] = nn.Dropout(dropout)

        if dropout is not None:
            del layers[list(layers.keys())[-1]]
        del layers[list(layers.keys())[-1]]
        self.net = nn.Sequential(layers)

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        (_, _, size) = adjs[-1]
        x_target = x[:size[1]]  # Target nodes are always placed first.
        x = self.net(x_target)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        return self.net(x_all)
