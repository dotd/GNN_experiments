import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Reddit, PPI, Amazon, Planetoid, Flickr, GitHub
from torch_geometric.utils import to_networkx
from os.path import join

from tst.ogb.main_pyg_with_pruning import get_args

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
    elif dataset_name == 'github':
        dataset = GitHub(path)
    elif dataset_name == 'ppi':
        dataset = PPI(path)
    elif dataset_name in ['amazon_comp', 'amazon_photo']:
        dataset = Amazon(path, "Computers", T.NormalizeFeatures()) if dataset_name == 'amazon_comp' else Amazon(path,
                                                                                                                "Photo",
                                                                                                                T.NormalizeFeatures())
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


def print_line(datasets, stats, stat_name):
    for dataset_name in datasets:
        print(f' & {stats[dataset_name][stat_name]}', end='')

    print(r' \\')


def main():
    global graph
    args = get_args()
    random = np.random.RandomState(0)
    dataset_stats = dict()
    dataset_names = ['github', 'reddit', 'amazon_photo', 'amazon_comp', 'Cora', 'CiteSeer', 'PubMed', 'ppi']
    stat_names = ['n_nodes', 'n_edges', 'n_classes', 'average_degree']
    printable_stat_names = [r'\# Nodes', r'\# Edges', r'\# Classes', r'Average degree']

    # a = get_dataset('github')
    # b = get_dataset('reddit')

    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name)
        graph = to_networkx(data=dataset.data)

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_classes = len(np.unique(dataset.data.y)) if dataset_name not in ['ppi'] else dataset.data.y.shape[1]
        average_degree = np.mean([v for x, v in graph.degree()])

        dataset_stats[dataset_name] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_classes': n_classes,
            'average_degree': int(average_degree)
        }

    for printable_stat_name, stat_name in zip(printable_stat_names, stat_names):
        print(printable_stat_name, end='')
        print_line(dataset_names, dataset_stats, stat_name)


if __name__ == '__main__':
    main()
