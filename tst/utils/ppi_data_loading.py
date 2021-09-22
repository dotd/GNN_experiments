# Implementation from https://github.com/gordicaleksa/pytorch-GAT

import json
import os
import zipfile

import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from tst.ogb.main_pyg_with_pruning import prune_dataset

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')


def count_edges(edge_index_list):
    """
    Counts the number of edges in the edge index list
    Args:
        edge_index_list: a python list containing tensors describing the edge indices
    """
    return sum([edge_index.shape[1] for edge_index in edge_index_list])


def construct_pyg_dataset(node_features, node_labels, edge_index):
    """
    Constructs a list of PyG data objects from the given arguments
    Args:
        node_features: a list of node features for each graph
        node_labels: a list of node labels for each graph
        edge_index: a list of edge indices for each graph
    Returns list(Data): a list of pytorch-geometric data objects
    """
    return [Data(x=nodes, edge_index=edges, y=labels)
            for nodes, labels, edges in zip(node_features, node_labels, edge_index)]


def unstruct_pyg_dataset(dataset):
    """
    Extracts the parameters of a list of PyG data objects into separate lists of attributes
    Args:
        dataset: a list of Data objects
    """
    node_features, node_labels, edge_index = [], [], []
    for data in dataset:
        node_features.append(data.x)
        node_labels.append(data.y)
        edge_index.append(data.edge_index)

    return node_features, node_labels, edge_index


def load_graph_data(args, device, PPI_URL='https://data.dgl.ai/dataset/ppi.zip'):
    # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed
    if not os.path.exists(PPI_PATH):  # download the first time this is ran
        os.makedirs(PPI_PATH)

        # Step 1: Download the ppi.zip (contains the PPI dataset)
        zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
        download_url_to_file(PPI_URL, zip_tmp_path)

        # Step 2: Unzip it
        with zipfile.ZipFile(zip_tmp_path) as zf:
            zf.extractall(path=PPI_PATH)
        print(f'Unzipping to: {PPI_PATH} finished.')

        # Step3: Remove the temporary resource file
        os.remove(zip_tmp_path)
        print(f'Removing tmp file {zip_tmp_path}.')

    # Collect train/val/test graphs here
    edge_index_list = []
    node_features_list = []
    node_labels_list = []

    # Dynamically determine how many graphs we have per split (avoid using constants when possible)
    num_graphs_per_split_cumulative = [0]

    # Small optimization "trick" since we only need test in the playground.py
    splits = ['test'] if args.ppi_load_test_only else ['train', 'valid', 'test']

    for split in splits:
        # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
        # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
        # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
        # Note: node features are already preprocessed
        node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

        # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
        # SHAPE = (NS, 121)
        node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

        # Graph topology stored in a special nodes-links NetworkX format
        nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
        # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
        # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
        # because of the edge index and the way GAT implementation #3 works
        collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
        # For each node in the above collection, ids specify to which graph the node belongs to
        graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
        num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

        # Split the collection of graphs into separate PPI graphs
        for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
            mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
            graph_node_ids = np.asarray(mask).nonzero()[0]
            graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
            print(f'Loading {split} graph {graph_id} to CPU. '
                  f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

            # shape = (2, E) - where E is the number of edges in the graph
            # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
            # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
            edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
            edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
            edge_index_list.append(edge_index)
            # shape = (N, 50) - where N is the number of nodes in the graph
            node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
            # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))

    # Optimization, do a shortcut in case we only need the test data loader
    if args.ppi_load_test_only:
        data_loader_test = GraphDataLoader(
            node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            batch_size=args.batch_size,
            shuffle=False
        )
        return data_loader_test
    else:

        old_edge_count = count_edges(edge_index_list)
        train_dataset = construct_pyg_dataset(
            node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]])

        validation_dataset = construct_pyg_dataset(
            node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]])

        test_dataset = construct_pyg_dataset(
            node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]])

        pruning_params, prunning_ratio = prune_dataset(train_dataset, args, pruning_params=None)
        prune_dataset(validation_dataset, args, pruning_params=pruning_params)
        prune_dataset(test_dataset, args, pruning_params=pruning_params)

        train_node_features, train_node_labels, train_edge_index = unstruct_pyg_dataset(train_dataset)
        val_node_features, val_node_labels, val_edge_index = unstruct_pyg_dataset(validation_dataset)
        test_node_features, test_node_labels, test_edge_index = unstruct_pyg_dataset(test_dataset)

        edge_count = count_edges(train_edge_index + val_edge_index + test_edge_index)
        prune_ratio = edge_count / old_edge_count
        print(f"Old number of edges: {old_edge_count}. New one: {edge_count}. Change: {prune_ratio * 100}\%")

        data_loader_train = GraphDataLoader(
            train_node_features,
            train_node_labels,
            train_edge_index,
            batch_size=args.batch_size,
            shuffle=True
        )

        data_loader_val = GraphDataLoader(
            val_node_features,
            val_node_labels,
            val_edge_index,
            batch_size=args.batch_size,
            shuffle=False  # no need to shuffle the validation and test graphs
        )

        data_loader_test = GraphDataLoader(
            test_node_features,
            test_node_labels,
            test_edge_index,
            batch_size=args.batch_size,
            shuffle=False
        )

        return data_loader_train, data_loader_val, data_loader_test, prune_ratio


class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).
    """

    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it
    """

    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index


def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data
