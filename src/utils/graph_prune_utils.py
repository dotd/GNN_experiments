import numpy as np
import torch
from tqdm import tqdm


def get_adjacent_edges_of_nodes(num_nodes, edge_index, edge_attr):
    edge_index_from = edge_index[0, :]
    edge_index_to = edge_index[1, :]
    adjacent_nodes = [[] for _ in range(num_nodes)]
    adjacent_edges_features = [[] for _ in range(num_nodes)] if edge_attr is not None else None

    for i in range(len(edge_index_from)):
        # if (i / len(edge_index_from)) > 0.7:
        #     break
        if i % 1e+6 == 0 and i != 0:
            print(f"Processed {i} / {len(edge_index_from)} edges ({(i / len(edge_index_from)) * 100} %)")

        from_ = edge_index_from[i]
        to_ = edge_index_to[i]
        adjacent_nodes[from_].append(to_)

        if adjacent_edges_features is not None:
            adjacent_edges_features[from_].append(edge_attr[i])

    if adjacent_edges_features is not None:
        for i in range(len(adjacent_edges_features)):
            adjacent_edges_features[i] = torch.stack(adjacent_edges_features[i]) if len(adjacent_nodes[i]) != 0 else torch.tensor([])

    # adjacent_nodes = [edge_index_to[edge_index_from == i] for i in range(num_nodes)]
    #
    # if len(edge_attr.shape) == 1:
    #     adjacent_edges_features = [edge_attr[edge_index_from == i] for i in range(num_nodes)] if edge_attr is not None else None
    # else:
    #     adjacent_edges_features = [edge_attr[edge_index_from == i, :] for i in range(num_nodes)] if edge_attr is not None else None

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


def chunk_string(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]


def _prune_edges_by_minhash_lsh_helper(num_nodes,
                                       edge_list,
                                       edge_attrs,
                                       node_attr,
                                       minhash,
                                       lsh_nodes,
                                       lsh_edges,
                                       prunning_mode="all",
                                       complement=False,
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

    lsh_nodes_signatures = lsh_nodes.sign_vectors(node_attr) if lsh_nodes is not None else None

    for n in range(num_nodes):
        # Get all the adjacent nodes and edge attributes
        adjacent_nodes_local = adjacent_nodes[n]

        # If no adjacent nodes, let's continue
        if len(adjacent_nodes_local) == 0:
            continue

        # Set of all adjacent nodes where we use the lsh representation for the set

        # Going over the adjacent for each node and make the mappings.
        if lsh_edges is not None:
            edge_attrs = adjacent_edges_attrs[n]
            adjacent_meta = [((n, node.item()), edge_attrs[idx]) for idx, node in enumerate(adjacent_nodes_local)]
            signatures_edge_attrs = lsh_edges.sign_vectors(edge_attrs.numpy()) if lsh_edges is not None else None
        else:
            adjacent_meta = [((n, node.item()), node_attr[node.item()]) for node in adjacent_nodes_local]

        if lsh_nodes is not None and lsh_edges is not None:
            source_tensor = lsh_nodes_signatures[n][np.newaxis, ...].repeat(repeats=len(adjacent_nodes_local), axis=0)
            target_tensor = np.stack([lsh_nodes_signatures[adj] for adj in adjacent_nodes_local])
            rep_tensor = np.hstack([source_tensor, signatures_edge_attrs, target_tensor])
        elif lsh_nodes is None:
            rep_tensor = signatures_edge_attrs
        elif lsh_edges is None:
            source_tensor = lsh_nodes_signatures[n][np.newaxis, ...].repeat(repeats=len(adjacent_nodes_local), axis=0)
            target_tensor = np.stack([lsh_nodes_signatures[adj] for adj in adjacent_nodes_local])
            rep_tensor = np.hstack([source_tensor, target_tensor])
        else:
            raise Exception("No features in the graph")

        rep_dim = rep_tensor.shape[-1]
        rep_flat_str = ''.join(map(str, rep_tensor.flatten()))
        adjacent_reps = chunk_string(rep_flat_str, rep_dim)

        # Transform the adjacent nodes to their signatures
        results = minhash.apply(adjacent_reps, adjacent_meta)
        results_metas = [result.meta for result in results]
        complement_results_metas = [meta for meta in adjacent_meta if meta not in results_metas]
        # The pruned list construction
        if complement:
            results_metas = complement_results_metas
        for meta in results_metas:
            new_edges_list.append(meta[0])
            new_attr_list.append(torch.tensor(meta[1]))
    # We return a numpy array
    new_edges_torch = torch.LongTensor(new_edges_list).T
    new_attr_torch = torch.stack(new_attr_list, axis=0) if len(new_attr_list) else torch.tensor([])

    return new_edges_torch, new_attr_torch


def dataset_prune_edges_by_minhash_lsh(graph_dataset, minhash, lsh):
    for i, graph_sample in enumerate(graph_dataset.samples):
        new_edges = graph_prune_edges_by_minhash_lsh(graph_sample, minhash, lsh)
        graph_sample.set_edges_list(new_edges)


def tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh_nodes, lsh_edges, complement):
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
                                                             lsh_edges=lsh_edges,
                                                             complement=complement, )
    tg_sample.edge_index = new_edges
    if old_edge_attr is not None and len(old_edge_attr.shape) == 1 and len(new_attr.shape) == 2 and new_attr.shape[1] == 1:
        tg_sample.edge_attr = new_attr.squeeze()
    else:
        tg_sample.edge_attr = new_attr


def tg_dataset_prune_edges_by_minhash_lsh(tg_dataset, minhash, lsh_nodes, lsh_edges, complement):
    ratios = list()
    for tg_sample in tqdm(tg_dataset):
        original_number_of_edges = tg_sample.edge_index.shape[1]
        tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh_nodes, lsh_edges, complement)

        if original_number_of_edges != 0:
            new_number_of_edges = tg_sample.edge_index.shape[1]
            ratios.append(new_number_of_edges / original_number_of_edges)
        else:
            ratios.append(0)
    return np.mean(ratios)


def tg_sample_prune_random(tg_sample, p, random):
    num_edges = tg_sample.edge_index.shape[1]
    index_p = int(p * num_edges)
    indices = random.permutation(num_edges)[:index_p]
    tg_sample.edge_index = tg_sample.edge_index[:, indices]
    if hasattr(tg_sample, 'edge_attr') and tg_sample.edge_attr is not None:
        if len(tg_sample.edge_attr.shape) == 1:
            tg_sample.edge_attr = tg_sample.edge_attr[indices]
        else:
            tg_sample.edge_attr = tg_sample.edge_attr[indices, :]


def tg_dataset_prune_random(tg_dataset, p, random):
    for tg_sample in tqdm(tg_dataset):
        tg_sample_prune_random(tg_sample, p, random)


def tg_dataset_prune(tg_dataset, method, complement, **kwargs):
    if method == "minhash_lsh":
        prunning_ratio = tg_dataset_prune_edges_by_minhash_lsh(tg_dataset,
                                                               minhash=kwargs.get("minhash"),
                                                               lsh_nodes=kwargs.get("lsh_nodes"),
                                                               lsh_edges=kwargs.get("lsh_edges"),
                                                               complement=complement, )

        return prunning_ratio
    if method == "random":
        tg_dataset_prune_random(tg_dataset, p=kwargs.get("p"), random=kwargs.get("random"))
