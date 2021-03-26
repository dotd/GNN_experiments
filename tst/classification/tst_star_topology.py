import numpy as np
import torch

from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHashRep
# The function to test...
from src.utils.graph_prune_utils import _prune_edges_by_minhash_lsh_helper


def tst_minhash_lsh_star(random=np.random.RandomState(0)):
    star_degree = 5
    dim_nodes = 4
    dim_edges = 3
    # around is indices 0 to (star_degree-1) inclusive while center is star_degree
    num_nodes = 1 + star_degree
    # indices correspond to nodes
    num_edges = star_degree

    # MinHash parameters
    num_minhash_funcs = 1
    minhash = MinHashRep(num_minhash_funcs, random, prime=2147483647)
    print(f"minhash:\n{minhash}")

    # LSH parameters
    lsh_num_funcs = 2
    sparsity = 2
    std_of_threshold = 1
    lsh_nodes = LSH(dim_nodes,
                    num_functions=lsh_num_funcs,
                    sparsity=sparsity,
                    std_of_threshold=std_of_threshold,
                    random=random)
    print(f"lsh_nodes:\n{lsh_nodes}")

    lsh_edges = LSH(dim_edges,
                    num_functions=lsh_num_funcs,
                    sparsity=sparsity,
                    std_of_threshold=std_of_threshold,
                    random=random)
    print(f"lsh_edges:\n{lsh_edges}")

    # Topology of edges: Shape is [2,num edges], type torch.int64
    topology_list = [[star_degree, i] for i in range(star_degree)]
    edge_index = torch.tensor(topology_list, dtype=torch.int64).T
    print(f"edge_index=\n{edge_index}")

    # edge features
    attr_numpy = random.normal(size=(num_edges, dim_edges))
    edge_attr = torch.tensor(attr_numpy.tolist(), dtype=torch.float32)
    print(f"edge_attr=\n{edge_attr}")

    # node features
    node_attr = random.normal(size=(num_nodes, dim_nodes))
    print(f"node_attr=\n{node_attr}")

    # Do the prunning
    new_edges, new_attr = _prune_edges_by_minhash_lsh_helper(num_nodes,
                                                             edge_list=edge_index,
                                                             edge_attrs=edge_attr,
                                                             node_attr=node_attr,
                                                             minhash=minhash,
                                                             lsh_nodes=lsh_nodes,
                                                             lsh_edges=None)


if __name__ == "__main__":
    tst_minhash_lsh_star()
