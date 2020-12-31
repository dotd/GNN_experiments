import numpy as np
import time
start_time = time.time()

import torch
from torch_geometric.data import DataLoader

from src.synthetic.random_graph_dataset import generate_graphs_dataset
from tst.torch_geometric.tst_torch_geometric1 import GCN
from src.synthetic.synthetic_utils import transform_dataset_to_torch_geometric_dataset
from tst.torch_geometric.tst_torch_geometric1 import train, func_test


from src.utils.graph_prune_utils import graph_prune_edges_by_minhash_lsh
from src.utils.graph_prune_utils import dataset_prune_edges_by_minhash_lsh
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash


def tst_classify_synthetic():
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")

    random = np.random.RandomState(0)

    # Dataset parameters
    num_samples = 1000
    num_classes = 2
    min_nodes = 10
    max_nodes = 10
    dim_nodes = 4
    noise_nodes = 1
    connectivity_rate = 0.2
    connectivity_rate_noise = 0.05
    symmetric_flag = True
    noise_remove_node = 0.1
    noise_add_node = 0.1

    graph_dataset = generate_graphs_dataset(num_samples=num_samples,
                                            num_classes=num_classes,
                                            min_nodes=min_nodes,
                                            max_nodes=max_nodes,
                                            dim_nodes=dim_nodes,
                                            noise_nodes=noise_nodes,
                                            connectivity_rate=connectivity_rate,
                                            connectivity_rate_noise=connectivity_rate_noise,
                                            noise_remove_node=noise_remove_node,
                                            noise_add_node=noise_add_node,
                                            symmetric_flag=symmetric_flag,
                                            random=random)
    # MinHash parameters
    num_minhash_funcs = 2
    minhash = MinHash(num_minhash_funcs, random, prime=2147483647)
    print(f"minhash:\n{minhash}")

    # LSH parameters
    lsh_num_funcs = 2
    sparsity = 3
    std_of_threshold = 1
    lsh = LSH(dim_nodes,
              num_functions=lsh_num_funcs,
              sparsity=sparsity,
              std_of_threshold=std_of_threshold,
              random=random)
    print(f"lsh:\n{lsh}")

    # Prune
    dataset_prune_edges_by_minhash_lsh(graph_dataset, minhash, lsh)

    print(f"{time.time() - start_time:.4f} Finished generating dataset")

    # print("")
    # print(graph_dataset)
    tg_dataset = transform_dataset_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels, num_classes)
    train_loader = DataLoader(tg_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(tg_dataset, batch_size=64, shuffle=False)
    model = GCN(hidden_channels=60, in_size=dim_nodes, out_size=num_classes)
    train(model, train_loader)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    for epoch in range(10):
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    print(f"{time.time() - start_time:.4f} start time")
    tst_classify_synthetic()
