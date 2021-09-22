from multiprocessing.pool import ThreadPool
from os.path import join

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Reddit, PPI, Amazon, Planetoid, Flickr, GitHub, ZINC, QM9
from torch_geometric.utils import to_networkx

from src.utils.graph_prune_utils import tg_dataset_prune
from tst.ogb.main_pyg_with_pruning import get_args
from tst.ogb.main_with_pruning_node_prediction import get_dataset

graph = None


def get_dataset(dataset_name):
    """
    Retrieves the dataset corresponding to the given name.
    """
    path = join('dataset', dataset_name)
    if dataset_name == 'reddit':
        dataset = Reddit(path)
    elif dataset_name == 'flickr':
        dataset = Flickr(path)
    elif dataset_name == 'zinc':
        dataset = ZINC(root='dataset', subset=True, split='train')
    elif dataset_name == 'QM9':
        dataset = QM9(root='dataset')
    elif dataset_name == 'github':
        dataset = GitHub(path)
    elif dataset_name == 'ppi':
        dataset = PPI(path)
    elif dataset_name in ['amazon_comp', 'amazon_photo']:
        dataset = Amazon(path, "Computers", T.NormalizeFeatures()) if dataset_name == 'amazon_comp' else Amazon(path, "Photo", T.NormalizeFeatures())
        data = dataset.data
        idx_train, idx_test = train_test_split(list(range(data.x.shape[0])), test_size=0.4, random_state=42)
        idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)
        data.train_mask = torch.tensor(idx_train)
        data.val_mask = torch.tensor(idx_val)
        data.test_mask = torch.tensor(idx_test)
        dataset.data = data
    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name=dataset_name, split="public", transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError

    return dataset


def get_neighborhood_size_for_node(node, k):
    return len(nx.single_source_shortest_path_length(graph, node, cutoff=k))


def get_k_hop_neighborhood_size_variance_for(k):
    with ThreadPool(processes=20) as pool:
        args = ((node, k) for node in graph.nodes)
        results = pool.starmap(get_neighborhood_size_for_node, args)

    return np.var(results)


def main():
    global graph
    args = get_args()
    random = np.random.RandomState(0)

    for k in range(1, 11, 2):
        print(f"=========== K-hop neighborhood sizes for k={k} ===========")
        for p in np.arange(0.1, 1.1, 0.1):

            dataset = get_dataset(args.dataset)
            tg_dataset_prune(tg_dataset=[dataset.data],
                             method="random",
                             p=p,
                             random=random,
                             complement=False)
            graph = to_networkx(data=dataset.data)
            var = get_k_hop_neighborhood_size_variance_for(k)
            print(f"({p}, {var})")


if __name__ == '__main__':
    main()
