import numpy as np
import torch
from tqdm import tqdm


def get_adjacent_edges_of_nodes(num_nodes, edge_index, edge_attr):
    num_edges = edge_index.shape[1]
    adjacent_nodes = dict()
    adjacent_edges_features = dict()

    # Initialization
    for i in range(num_nodes):
        adjacent_nodes[i] = list()
        adjacent_edges_features[i] = list()

    # Doing the work: going for each edge and put it in the right map of edges and features
    for i in range(num_edges):
        edge_from = edge_index[0, i].item()
        edge_to = edge_index[1, i].item()
        adjacent_nodes[edge_from].append(edge_to)
        if edge_attr is not None:
            edge_feature = edge_attr[i, :]
            adjacent_edges_features[edge_from].append(edge_feature)

    return adjacent_nodes, adjacent_edges_features


def graph_prune_edges_by_minhash_lsh(graph_sample, minhash, lsh):
    vectors = graph_sample.nodes_vecs
    edges_list = graph_sample.get_edges_list()
    num_nodes = graph_sample.num_nodes
    new_edges = _prune_edges_by_minhash_lsh_helper(num_nodes,
                                                   edges_list,
                                                   vectors,
                                                   minhash,
                                                   lsh)
    return new_edges


def _prune_edges_by_minhash_lsh_helper(num_nodes,
                                       edge_list,
                                       edge_attrs,
                                       node_attr,
                                       minhash,
                                       lsh_nodes,
                                       lsh_edges,
                                       prunning_mode="all"
                                       ):
    """
    :param num_nodes:
    :param edge_list:
    :param edge_attrs:
    :param node_attr:
    :param minhash:
    :param lsh:
    :param prunning_mode: There is "all" for node_attr & edge_attr, "node" for only local nodes, and "edge" for only adjacent edges
    :return:
    """
    # Num edges
    num_edges = edge_list.shape[1]
    # The pruned list of edges
    new_edges_list = list()
    new_attr_list = list()

    # for each node get a list of adjacent nodes and the corresponding representations.
    adjacent_nodes, adjacent_edges_attrs = get_adjacent_edges_of_nodes(num_nodes,
                                                                       edge_list,
                                                                       edge_attrs)
    lsh_nodes_signatures = list()
    lsh_nodes_signatures_str = list()
    if lsh_nodes is not None:
        for n in range(num_nodes):
            vector = node_attr[n, :]
            signature = lsh_nodes.sign_vector(vector)
            lsh_nodes_signatures.append(signature)
            lsh_nodes_signatures_str.append("".join(f"{x}" for x in signature))

    for n in range(num_nodes):
        # Get all the adjacent nodes and edge attributes
        adjacent_nodes_local = adjacent_nodes[n]
        edge_attrs = adjacent_edges_attrs[n]

        # If no adjacent nodes, let's continue
        if len(adjacent_nodes_local) == 0:
            continue

        # Set of all adjacent nodes where we use the lsh representation for the set
        adjacent_reps = list()
        adjacent_meta = list()

        # Going over the adjacent for each node and make the mappings.
        for idx, node in enumerate(adjacent_nodes_local):
            rep = ""
            edge_attr = edge_attrs[idx]
            if lsh_nodes is not None:
                rep += lsh_nodes_signatures_str[node]

            rep += "_"

            if lsh_edges is not None:
                signature_edge_attr = lsh_edges.sign_vector(edge_attr.numpy())
                signature_edge_attr_str = "".join(f"{x}" for x in signature_edge_attr)
                rep += signature_edge_attr_str
            adjacent_reps.append(rep)
            adjacent_meta.append(((n, node), edge_attr))

        # Transform the adjacent nodes to their signatures
        results = minhash.apply(adjacent_reps, adjacent_meta)
        # The pruned list construction
        for result in results:
            new_edges_list.append(result.meta[0])
            new_attr_list.append(result.meta[1])

    # We return a numpy array
    new_edges_torch = torch.LongTensor(new_edges_list).T
    new_attr_torch = torch.stack(new_attr_list, axis=0)
    return new_edges_torch, new_attr_torch


def dataset_prune_edges_by_minhash_lsh(graph_dataset, minhash, lsh):
    for i, graph_sample in enumerate(graph_dataset.samples):
        new_edges = graph_prune_edges_by_minhash_lsh(graph_sample, minhash, lsh)
        graph_sample.set_edges_list(new_edges)


def tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh_nodes, lsh_edges):
    # Get the device so we know to where to device later
    device = tg_sample.edge_index.get_device()

    # Get the edges as torch and numpy
    old_edge_index = tg_sample.edge_index
    old_edge_attr = tg_sample.edge_attr

    # Get node features and num of nodes.
    num_nodes = tg_sample.x.shape[0]
    old_x_numpy = tg_sample.x.numpy()

    # Do the prunning
    new_edges, new_attr = _prune_edges_by_minhash_lsh_helper(num_nodes,
                                                             edge_list=old_edge_index,
                                                             edge_attrs=old_edge_attr,
                                                             node_attr=old_x_numpy,
                                                             minhash=minhash,
                                                             lsh_nodes=lsh_nodes,
                                                             lsh_edges=lsh_edges)
    tg_sample.edge_index = new_edges
    tg_sample.edge_attr = new_attr


def tg_dataset_prune_edges_by_minhash_lsh(tg_dataset, minhash, lsh_nodes, lsh_edges):
    ratios = list()
    bar = tqdm(total=len(tg_dataset))
    for i, tg_sample in enumerate(tg_dataset):
        original_number_of_edges = tg_sample.edge_index.shape[1]
        tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh_nodes, lsh_edges)
        new_number_of_edges = tg_sample.edge_index.shape[1]
        ratios.append(new_number_of_edges / original_number_of_edges)
        bar.update(1)
    return np.mean(ratios)


def tg_sample_prune_random(tg_sample, p, random):
    num_edges = tg_sample.edge_index.shape[1]
    index_p = int(p * num_edges)
    indices = random.permutation(num_edges)[:index_p]
    tg_sample.edge_index = tg_sample.edge_index[:, indices]
    if hasattr(tg_sample, 'edge_attr') and tg_sample.edge_attr is not None:
        tg_sample.edge_attr = tg_sample.edge_attr[indices, :]


def tg_dataset_prune_random(tg_dataset, p, random):
    bar = tqdm(total=len(tg_dataset))
    for i, tg_sample in enumerate(tg_dataset):
        tg_sample_prune_random(tg_sample, p, random)
        bar.update(1)


def tg_dataset_prune(tg_dataset, method, **kwargs):
    if method == "minhash_lsh":
        prunning_ratio = tg_dataset_prune_edges_by_minhash_lsh(tg_dataset,
                                                               minhash=kwargs.get("minhash"),
                                                               lsh_nodes=kwargs.get("lsh_nodes"),
                                                               lsh_edges=kwargs.get("lsh_edges"))
        return prunning_ratio
    if method == "random":
        tg_dataset_prune_random(tg_dataset, p=kwargs.get("p"), random=kwargs.get("random"))
