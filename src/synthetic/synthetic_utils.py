import torch
import torch_geometric as tg
from torch_geometric.utils.convert import from_networkx


def transform_networkx_sample_to_torch_geometric_data(networkx_sample, label):
    graph_tg = from_networkx(networkx_sample)
    graph_tg.y = torch.tensor([label])
    return graph_tg


def transform_graph_sample_to_torch_geometric_data(graph_sample, label):
    x = torch.from_numpy(graph_sample.nodes_vecs).type(torch.FloatTensor)
    edge_index = torch.from_numpy(graph_sample.get_edges_list()).type(torch.LongTensor)
    label_tensor = torch.tensor([label])
    graph_tg = tg.data.Data(x=x, edge_index=edge_index, y=label_tensor)
    return graph_tg


def transform_networkx_to_torch_geometric_dataset(networkx_samples, labels):
    tg_dataset = list()
    for idx in range(len(networkx_samples)):
        network_sample = networkx_samples[idx]
        label = labels[idx]
        tg_dataset.append(transform_networkx_sample_to_torch_geometric_data(network_sample, label))
    return tg_dataset


def transform_dataset_to_torch_geometric_dataset(graph_samples, labels):
    tg_dataset = list()
    for idx in range(len(graph_samples)):
        graph_sample = graph_samples[idx]
        label = labels[idx]
        tg_dataset.append(transform_graph_sample_to_torch_geometric_data(graph_sample, label))
    return tg_dataset


