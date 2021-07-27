from collections import OrderedDict

import numpy as np
from torch import nn


class MLP(nn.Module):
    """
    A module which consists of several n_steps fully connected layers with ReLU activations
    between every 2 layers. The output of the module has no activation
    """
    def __init__(self, in_features, out_features, n_steps=6, dropout=0.7):
        super(MLP, self).__init__()
        # steps = np.geomspace(in_features, out_features, n_steps + 1, dtype=int)
        steps = np.linspace(in_features, out_features, n_steps + 1, dtype=int)
        steps[0] = in_features
        steps[-1] = out_features
        layers = OrderedDict()
        for idx, (in_f, out_f) in enumerate(zip(steps[: -1], steps[1:])):
            layers[f"linear_{idx}"] = nn.Linear(in_features=in_f, out_features=out_f)
            layers[f"relu_{idx}"] = nn.ELU()
            if dropout is not None:
                layers[f"dropout_{idx}"] = nn.Dropout(dropout)

        if dropout is not None:
            del layers[list(layers.keys())[-1]]
        del layers[list(layers.keys())[-1]]
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)
