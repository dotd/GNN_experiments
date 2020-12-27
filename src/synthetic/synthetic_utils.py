import torch
import torch_geometric as tg


def transform_graph_sample_to_torch_geometric_data(graph_sample):
    # graph_tg = tg.data.Data(x=node_features_tensor, edge_index=edge_index_tensor)
    x = torch.from_numpy(graph_sample.nodes_vecs)
    edge_index = torch.from_numpy(graph_sample.get_edges_list())

    graph_tg = tg.data.Data(x=graph_sample.nodes_vecs, edge_index=edge_index)
    return graph_tg


def transform_dataset_to_torch_geometric_dataset(dataset):
    tg_dataset = list()
    for graph_sample in range(dataset):
        tg_dataset.append(transform_graph_sample_to_torch_geometric_data(graph_sample))
    return tg_dataset