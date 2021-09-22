import torch
import torch_geometric as tg
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm


def transform_networkx_sample_to_torch_geometric_data(networkx_sample, label):
    graph_tg = from_networkx(networkx_sample)
    graph_tg.y = torch.tensor([label])
    return graph_tg


def transform_graph_sample_to_torch_geometric_data(graph_sample, label):
    x = torch.from_numpy(graph_sample.nodes_vecs).type(torch.FloatTensor)
    edge_index = torch.from_numpy(graph_sample.get_edges_list()).type(torch.LongTensor)
    edge_attr = torch.from_numpy(graph_sample.edges_vecs).type(torch.FloatTensor)
    label_tensor = torch.tensor([label])
    graph_tg = tg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label_tensor)
    return graph_tg


def transform_networkx_to_torch_geometric_dataset(networkx_samples, labels):
    print("Transforming networkx samples to PyG")
    tg_dataset = [transform_networkx_sample_to_torch_geometric_data(networkx_sample=networkx_samples[idx],
                                                                    label=labels[idx])
                  for idx in tqdm(range(len(networkx_samples)))]
    return tg_dataset


def transform_dataset_to_torch_geometric_dataset(graph_samples, labels):
    print("Transforming graph samples samples to PyG")
    tg_dataset = list()
    for idx in tqdm(range(len(graph_samples))):
        graph_sample = graph_samples[idx]
        label = labels[idx]
        tg_dataset.append(transform_graph_sample_to_torch_geometric_data(graph_sample, label))
    return tg_dataset

