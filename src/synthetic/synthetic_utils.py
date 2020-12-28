import torch
import torch_geometric as tg
import torch.nn.functional as F


def transform_graph_sample_to_torch_geometric_data(graph_sample, label, num_classes):
    # graph_tg = tg.data.Data(x=node_features_tensor, edge_index=edge_index_tensor)
    x = torch.from_numpy(graph_sample.nodes_vecs).type(torch.FloatTensor)
    edge_index = torch.from_numpy(graph_sample.get_edges_list()).type(torch.LongTensor)
    label_tensor = torch.tensor([label])
    graph_tg = tg.data.Data(x=x, edge_index=edge_index, y=label_tensor)
    return graph_tg


def transform_dataset_to_torch_geometric_dataset(graph_samples, labels, num_classes):
    tg_dataset = list()
    for idx in range(len(graph_samples)):
        graph_sample = graph_samples[idx]
        label = labels[idx]
        tg_dataset.append(transform_graph_sample_to_torch_geometric_data(graph_sample, label, num_classes))
    return tg_dataset
