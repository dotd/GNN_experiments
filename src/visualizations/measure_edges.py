import numpy as np

from src.utils.graph_prune_utils import tg_dataset_prune
from src.visualizations.neighborhood_sizes import get_dataset
from tst.ogb.main_pyg_with_pruning import get_args

graph = None


def count_edges(graphs):
    return sum([g.num_edges for g in graphs])


def remove_edges(graphs, k=1):
    for graph in graphs:
        all_edges = np.array([])
        for node in range(graph.num_nodes):
            edges, = np.where(graph.edge_index[0, :] == node)
            edges = edges[:k]
            all_edges = np.concatenate([all_edges, edges])
            pass
        graph.edge_index = graph.edge_index[:, all_edges]
        graph.edge_attr = graph.edge_attr[all_edges]
    return graphs


def main():
    global graph
    args = get_args()
    args.dataset = 'zinc'

    for k in range(1, 10, 1):
        dataset = get_dataset(args.dataset)
        graphs = [g for g in dataset]
        orig_num_edges = count_edges(graphs)
        pruned_graph = remove_edges(graphs, k)
        new_num_edges = count_edges(graphs)
        print(f"{k}: {new_num_edges / orig_num_edges}")
        # graph = to_networkx(data=dataset.data)
        # new_graph
        # print(f"({p}, {var})")


if __name__ == '__main__':
    main()
