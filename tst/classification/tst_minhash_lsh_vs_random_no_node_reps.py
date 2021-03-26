import numpy as np
import time
import copy
start_time = time.time()

from torch_geometric.data import DataLoader

from src.synthetic.random_graph_dataset import generate_graphs_dataset, add_random_gaussian_edge_attr
from tst.torch_geometric.tst_torch_geometric1 import GCN
from src.synthetic.synthetic_utils import transform_dataset_to_torch_geometric_dataset
from tst.torch_geometric.tst_torch_geometric1 import train, func_test


from src.utils.graph_prune_utils import tg_dataset_prune
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash


def tst_minhash_lsh_vs_random(random=np.random.RandomState(0)):
    """
    This tst shows how to
    (1) Generate Dataset
    (2) Do the prunning
    (3) Do classification.
    :return:
    """
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")

    # (1) Generate Syhthetic dataset
    print("(1) Generate Syhthetic dataset")
    # Dataset parameters
    num_samples = 1000
    num_classes = 10
    min_nodes = 20
    max_nodes = 30
    dim_nodes = 0
    dim_edges = 2
    connectivity_rate = 0.2
    connectivity_rate_noise = 0.15
    symmetric_flag = True
    noise_remove_node = 0.0
    hidden_channels = 10
    nodes_order_scramble_flag = True
    centers_nodes_std = 1
    centers_edges_std = 1
    node_additive_noise_std = 0.1
    edge_additive_noise_std = 0.1

    epoch_times = 15

    graph_dataset = generate_graphs_dataset(num_samples=num_samples,
                                            num_classes=num_classes,
                                            min_nodes=min_nodes,
                                            max_nodes=max_nodes,
                                            dim_nodes=dim_nodes,
                                            dim_edges=dim_edges,
                                            connectivity_rate=connectivity_rate,
                                            connectivity_rate_noise=connectivity_rate_noise,
                                            noise_remove_node=noise_remove_node,
                                            node_additive_noise_std=node_additive_noise_std,
                                            edge_additive_noise_std=edge_additive_noise_std,
                                            symmetric_flag=symmetric_flag,
                                            centers_nodes_std=centers_nodes_std,
                                            centers_edges_std=centers_edges_std,
                                            nodes_order_scramble_flag=nodes_order_scramble_flag,
                                            random=random)

    print("Show centers")
    for idx, center in enumerate(graph_dataset.centers):
        print(f"center {idx}")
        print(f"{center.__str__()}")


    # Next, we define both Minhash and LSH for generating datasets
    # MinHash parameters
    num_minhash_funcs = 1
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

    # (2) Create the dataset
    print("(2) Create the dataset")
    # Transform the dataset
    tg_dataset_original = transform_dataset_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels)


    # (3) Do the prunning
    print("(3) Do the prunning")
    # Make a copy for comparison that the manipulation of the prunning worked.
    tg_dataset_minhash_lsh = copy.deepcopy(tg_dataset_original)
    tg_dataset_random = copy.deepcopy(tg_dataset_original)

    # Do the pruneing according to the two methods:
    prunning_ratio = tg_dataset_prune(tg_dataset_minhash_lsh, "minhash_lsh", minhash=minhash, lsh=lsh)
    print(f"prunning_ratio = {prunning_ratio}")
    tg_dataset_prune(tg_dataset_random, "random", p=prunning_ratio, random=random)

    # Show some samples:
    print("")
    for i in range(min(1, len(graph_dataset.samples))):
        print(f"{i}) Original=\n{tg_dataset_original[i].edge_index.shape}")
        print(f"{i}) Pruned minhash_lsh=\n{tg_dataset_minhash_lsh[i].edge_index.shape}")
        print(f"{i}) Pruned random=\n{tg_dataset_random[i].edge_index.shape}")

    print(f"{time.time() - start_time:.4f} Finished generating dataset")

    # (4) Do the training original.
    print("(4) Do the training original")
    train_loader = DataLoader(tg_dataset_original, batch_size=64, shuffle=True)
    test_loader = DataLoader(tg_dataset_original, batch_size=64, shuffle=False)
    model = GCN(hidden_channels=hidden_channels, in_size=dim_nodes, out_size=num_classes)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    epoch_times_original = list()
    for epoch in range(epoch_times):
        start_epoch = time.time()
        train(model, train_loader, lr=0.01)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        epoch_times_original.append(time.time() - start_epoch)
    test_acc_original = test_acc

    # (5) Do the training minhash_lsh.
    print("(5) Do the training minhash_lsh")
    train_loader = DataLoader(tg_dataset_minhash_lsh, batch_size=64, shuffle=True)
    test_loader = DataLoader(tg_dataset_minhash_lsh, batch_size=64, shuffle=False)
    model = GCN(hidden_channels=hidden_channels, in_size=dim_nodes, out_size=num_classes)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    epoch_times_minhash_lsh = list()
    for epoch in range(epoch_times):
        start_epoch = time.time()
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        epoch_times_minhash_lsh.append(time.time() - start_epoch)
    test_acc_minhash_lsh = test_acc

    # (6) Do the training random.
    print("(6) Do the training random")
    train_loader = DataLoader(tg_dataset_random, batch_size=64, shuffle=True)
    test_loader = DataLoader(tg_dataset_random, batch_size=64, shuffle=False)
    model = GCN(hidden_channels=hidden_channels, in_size=dim_nodes, out_size=num_classes)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    epoch_times_random = list()
    for epoch in range(epoch_times):
        start_epoch = time.time()
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        epoch_times_random.append(time.time() - start_epoch)
    test_acc_random = test_acc

    print(f"Summary")
    print(f"original epochs time mean: {np.mean(epoch_times_original)}")
    print(f"original epochs time minhash_lsh: {np.mean(epoch_times_minhash_lsh)}")
    print(f"original epochs time random: {np.mean(epoch_times_random)}")
    print(f"final test_acc_original: {test_acc_original}")
    print(f"final test_acc_minhash_lsh: {test_acc_minhash_lsh}")
    print(f"final test_acc_random: {test_acc_random}")


if __name__ == "__main__":
    print(f"{time.time() - start_time:.4f} start time")
    tst_minhash_lsh_vs_random()
