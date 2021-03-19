from functools import partial

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from tst.ogb.encoder_utils import ASTNodeEncoder, get_vocab_mapping
from tst.ogb.gcn import GCN


def get_output_dimension(dataset: PygGraphPropPredDataset):
    out_dim = None
    if dataset.name == 'ogbg-molhiv':
        out_dim = dataset.num_tasks
    elif dataset.name == 'ogbg-molpcba':
        out_dim = dataset.num_tasks
    elif dataset.name == 'ogbg-ppa':
        out_dim = dataset.num_classes
    elif dataset.name == 'ogbg-code2':
        out_dim = dataset.num_classes
    else:
        raise NotImplementedError(f"Settings for dataset {dataset.name} not implemented")

    return out_dim


def create_model(dataset: PygGraphPropPredDataset, emb_dim: int, dropout_ratio: float, device: str, num_layers: int,
                 max_seq_len: int, num_vocab: int) -> torch.nn.Module:
    """
    Create a GCN model for the given dataset
    :param dataset:
    :param emb_dim:
    :param dropout_ratio:
    :param device:
    :param num_layers:
    :param max_seq_len:
    :param num_vocab:
    :return:
    """
    print("creating a model for ", dataset.name)
    if dataset.name == "ogbg-molhiv" or dataset.name == 'ogbg-molpcba':
        node_encoder = AtomEncoder(emb_dim=emb_dim)
        edge_encoder_constrtuctor = BondEncoder
        print("Number of classes: ", dataset.num_tasks)
        model = GCN(
                    num_classes=get_output_dimension(dataset),
                    # num_classes=dataset.num_classes,
                    num_layer=num_layers,
                    emb_dim=emb_dim,
                    drop_ratio=dropout_ratio,
                    node_encoder=node_encoder,
                    edge_encoder_ctor=edge_encoder_constrtuctor).to(device)

    elif dataset.name == "ogbg-code":
        nodetypes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'typeidx2type.csv.gz')
        nodeattributes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'attridx2attr.csv.gz')
        node_encoder = ASTNodeEncoder(emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                      num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)
        split_idx = dataset.get_idx_split()
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
        edge_encoder_ctor = partial(torch.nn.Linear, 2)
        print(f"Multiclassification with {len(vocab2idx)} classes. Num labels per example: {max_seq_len}")
        model = GCN(num_classes=len(vocab2idx), max_seq_len=max_seq_len, node_encoder=node_encoder,
                    edge_encoder_ctor=edge_encoder_ctor, num_layer=num_layers, emb_dim=emb_dim,
                    drop_ratio=dropout_ratio).to(device)
    elif dataset.name == "ogbg-ppa":
        node_encoder = torch.nn.Embedding(1, emb_dim)
        edge_encoder_ctor = partial(torch.nn.Linear, 7)
        model = GCN(num_classes=dataset.num_tasks, num_layer=num_layers,
                    emb_dim=emb_dim, drop_ratio=dropout_ratio,
                    node_encoder=node_encoder, edge_encoder_ctor=edge_encoder_ctor).to(device)
    else:
        raise ValueError("Used an invalid dataset name")
    return model


def add_zeros(data: Data) -> Data:
    """
    A utility function for the ogbg-code dataset which doesn't have node features.
    :param data: a torch_gemoetric Data object representing representing a graph
    :return:
    """
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data