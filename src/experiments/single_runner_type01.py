import numpy as np
import time
start_time = time.time()

from torch_geometric.data import DataLoader

from src.synthetic.random_graph_dataset import generate_graphs_dataset
from tst.torch_geometric.tst_torch_geometric1 import GCN
from src.synthetic.synthetic_utils import transform_dataset_to_torch_geometric_dataset
from tst.torch_geometric.tst_torch_geometric1 import train, func_test

from src.utils.graph_prune_utils import dataset_prune_edges_by_minhash_lsh
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash


def single_runner_type1(seed,
                        common_params,
                        dataset_params,
                        minhash_params,
                        lsh_params,
                        model_params
                        ):
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")

    random = np.random.RandomState(seed)

    graph_dataset = generate_graphs_dataset(num_samples=dataset_params.num_samples,
                                            num_classes=dataset_params.num_classes,
                                            min_nodes=dataset_params.min_nodes,
                                            max_nodes=dataset_params.max_nodes,
                                            dim_nodes=common_params.dim_nodes,
                                            noise_nodes=dataset_params.noise_nodes,
                                            connectivity_rate=dataset_params.connectivity_rate,
                                            connectivity_rate_noise=dataset_params.connectivity_rate_noise,
                                            noise_remove_node=dataset_params.noise_remove_node,
                                            noise_add_node=dataset_params.noise_add_node,
                                            symmetric_flag=dataset_params.symmetric_flag,
                                            random=random)
    # MinHash parameters
    minhash = MinHash(minhash_params.num_minhash_funcs,
                      random,
                      prime=2147483647)
    print(f"minhash:\n{minhash}")

    # LSH parameters
    lsh = LSH(din=common_params.dim_nodes,
              num_functions=lsh_params.lsh_num_funcs,
              sparsity=lsh_params.sparsity,
              std_of_threshold=lsh_params.std_of_threshold,
              random=random)
    print(f"lsh:\n{lsh}")

    # Prune
    dataset_prune_edges_by_minhash_lsh(graph_dataset, minhash, lsh)

    print(f"{time.time() - start_time:.4f} Finished generating dataset")

    # print("")
    # print(graph_dataset)
    tg_dataset = transform_dataset_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels, num_classes)
    train_loader = DataLoader(tg_dataset,
                              batch_size=64,
                              shuffle=True)
    test_loader = DataLoader(tg_dataset,
                             batch_size=64,
                             shuffle=False)
    model = GCN(hidden_channels=model_params.hidden_channels,
                in_size=common_params.dim_nodes,
                out_size=dataset_params.num_classes)
    train(model, train_loader)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    for epoch in range(model_params.num_episodes):
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    results = dict()
    results["test_acc"] = test_acc


