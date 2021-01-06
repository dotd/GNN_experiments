import numpy as np
import torch


def get_edges_of_nodes(num_nodes, edges_list):
    res = dict()
    for i in range(num_nodes):
        res[i] = set()
    num_edges = edges_list.shape[1]
    for i in range(num_edges):
        i0 = edges_list[0, i]
        i1 = edges_list[1, i]
        res[i0].add(i1)
    return res


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
                                       edges_list,
                                       vectors,
                                       minhash,
                                       lsh):
    new_edges = list()

    node_to_nodes = get_edges_of_nodes(num_nodes, edges_list)
    lsh_vectors = list()
    lsh_vectors_str = list()
    for n in range(num_nodes):
        signature = lsh.sign_vector(vectors[n])
        lsh_vectors.append(signature)
        lsh_vectors_str.append("".join(f"{x}" for x in signature))

    for n in range(num_nodes):
        adjacent = list(node_to_nodes[n])
        if len(adjacent) == 0:
            continue
        set_of_n = list()
        rep2edge_number = dict()
        for a in adjacent:
            set_of_n.append(lsh_vectors_str[a])
            rep2edge_number[lsh_vectors_str[a]] = a
        vec, vec_vals, translation = minhash.apply(set_of_n)
        for vec_val in vec_vals:
            new_edges.append([n, rep2edge_number[vec_val]])
    new_edges = np.array(new_edges).T

    return new_edges


def dataset_prune_edges_by_minhash_lsh(graph_dataset, minhash, lsh):
    for i, graph_sample in enumerate(graph_dataset.samples):
        new_edges = graph_prune_edges_by_minhash_lsh(graph_sample, minhash, lsh)
        graph_sample.set_edges_list(new_edges)


def tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh):
    old_edges = tg_sample.edge_index.numpy()
    old_centers = tg_sample.x.numpy()
    num_nodes = tg_sample.x.shape[0]
    new_edges = _prune_edges_by_minhash_lsh_helper(num_nodes,
                                       edges_list=old_edges,
                                       vectors=old_centers,
                                       minhash=minhash,
                                       lsh=lsh)
    new_edges_long = torch.LongTensor(new_edges)
    tg_sample.edge_index = new_edges_long


def tg_dataset_prune_edges_by_minhash_lsh(tg_dataset, minhash, lsh):
    for i, tg_sample in enumerate(tg_dataset):
        tg_sample_prune_edges_by_minhash_lsh(tg_sample, minhash, lsh)

