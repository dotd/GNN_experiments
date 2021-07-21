from functools import partial

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import to_networkx, from_networkx

from src.archs.MXMNet import MXMNet
from src.archs.SimpleGCN import SimpleGCN
from src.archs.pna import PNA
from tst.ogb.encoder_utils import ASTNodeEncoder, get_vocab_mapping
from src.archs.gat_geometric import GAT
from tst.ogb.gcn import GCN
from tst.ogb.gnn import GNN
from src.archs.monet import MoNet


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


def create_model(dataset: PygGraphPropPredDataset,
                 emb_dim: int,
                 dropout_ratio: float,
                 device: str,
                 num_layers: int,
                 max_seq_len: int,
                 num_vocab: int,
                 model_type: str) -> torch.nn.Module:
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

    elif dataset.name == "ogbg-code2":
        nodetypes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'typeidx2type.csv.gz')
        nodeattributes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'attridx2attr.csv.gz')
        print(nodeattributes_mapping)
        node_encoder = ASTNodeEncoder(emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                      num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)
        split_idx = dataset.get_idx_split()
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
        edge_encoder_ctor = partial(torch.nn.Linear, 2)
        print(f"Multiclassification with {len(vocab2idx)} classes. Num labels per example: {max_seq_len}")
        model = GCN(num_classes=len(vocab2idx), max_seq_len=max_seq_len, node_encoder=node_encoder,
                    edge_encoder_ctor=edge_encoder_ctor, num_layer=num_layers, emb_dim=emb_dim,
                    drop_ratio=dropout_ratio).to(device)
    elif dataset.name in ["ogbg-ppa"]:
        # Multi-class classification
        if model_type == 'gin':
            model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=num_layers, emb_dim=emb_dim,
                        drop_ratio=dropout_ratio, virtual_node=False).to(device)
        elif model_type == 'gin-virtual':
            model = GNN(gnn_type='gin', num_class=dataset.num_classes, num_layer=num_layers, emb_dim=emb_dim,
                        drop_ratio=dropout_ratio, virtual_node=True).to(device)
        elif model_type == 'gcn':
            model = GNN(gnn_type='gcn', num_class=dataset.num_classes, num_layer=num_layers, emb_dim=emb_dim,
                        drop_ratio=dropout_ratio, virtual_node=False).to(device)
        elif model_type == 'gcn-virtual':
            model = GNN(gnn_type='gcn', num_class=dataset.num_classes, num_layer=num_layers, emb_dim=emb_dim,
                        drop_ratio=dropout_ratio, virtual_node=True).to(device)
        elif model_type == 'gat':
            model = GAT(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        else:
            raise ValueError('Invalid GNN type')
    elif dataset.name == 'mnist':
        if model_type == 'gcn':
            model = SimpleGCN(num_node_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        elif model_type == 'monet':
            model = MoNet(kernel_size=25).to(device)
    elif dataset.name == 'zinc':
        if model_type == 'gat':
            return GAT(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        elif model_type == 'gcn':
            model = SimpleGCN(num_node_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        elif model_type == 'pna':
            model = PNA().to(device)
    elif dataset.name == 'QM9':
        if model_type == 'mxmnet':
            model = MXMNet().to(device)
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


def to_line_graph(data: Data, directed: bool = True) -> Data:
    """
    Convert a graph G to its corresponding line-graph L(G)
    Args:
        data: a torch_gemoetric Data object representing representing a graph
        directed: whether the original graph is directed or undirected
    """
    original_edge_attrs = data.edge_attr
    original_edge_names = [(from_.item(), to_.item()) for from_, to_ in zip(data.edge_index[0, :], data.edge_index[1, :])]
    original_edge_to_attr = {e: attr for e, attr in zip(original_edge_names, original_edge_attrs)}
    ctor = nx.DiGraph if directed else nx.Graph
    G = to_networkx(data,
                    node_attrs=['x'],
                    edge_attrs=['edge_attr'],
                    to_undirected=not directed)
    line_graph = nx.line_graph(G, create_using=ctor)
    res_data = from_networkx(line_graph)

    # Copy original attribtues
    res_data.x = torch.stack([original_edge_to_attr[e] for e in line_graph.nodes])
    res_data.y = data.y
    return data


def compose(data: Data, transforms: list):
    """
    Performs all the transformations given in transform on the data
    """
    for transform in transforms:
        data = transform(data)

    return data
