import numpy as np

from src.synthetic.random_graph_dataset import generate_graphs_dataset


def tst_generate_graphs_dataset():
    num_samples = 100
    num_classes = 2
    min_nodes = 5
    max_nodes = 10
    dim_nodes = 4
    noise_nodes = 0.1
    connectivity_rate = 0.2
    connectivity_rate_noise = 0.05
    symmetric_flag = True
    random = np.random.RandomState(0)
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

    print("")
    print(graph_dataset)


def tst_generate_graphs_dataset2():
    num_samples = 4
    num_classes = 2
    min_nodes = 4
    max_nodes = 5
    dim_nodes = 4
    dim_edges = 3
    connectivity_rate = 0.2
    connectivity_rate_noise = 0.05
    symmetric_flag = True
    random = np.random.RandomState(0)
    noise_remove_node = 0.1
    node_additive_noise_std = 0.01
    edge_additive_noise_std = 0.01
    centers_nodes_std = 1
    centers_edges_std = 1
    nodes_order_scramble_flag = False

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

    print("")
    print(graph_dataset)


if __name__ == "__main__":
    tst_generate_graphs_dataset2()
