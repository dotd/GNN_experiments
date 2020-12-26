import numpy as np

from src.synthetic.random_graph_dataset import generate_graphs_dataset


def tst_generate_graphs_dataset():
    num_samples = 5
    num_classes = 2
    min_nodes = 5
    max_nodes = 10
    dim_nodes = 4
    class_nodes_var = 0.1
    noise_nodes = 0.1
    connectivity_rate = 0.5
    connectivity_rate_noise = 0.05
    symmetric_flag = True
    random = np.random.RandomState(0)

    dataset = generate_graphs_dataset(num_samples=num_samples,
                                      num_classes=num_classes,
                                      min_nodes=min_nodes,
                                      max_nodes=max_nodes,
                                      dim_nodes=dim_nodes,
                                      class_nodes_var=class_nodes_var,
                                      noise_nodes=noise_nodes,
                                      connectivity_rate=connectivity_rate,
                                      connectivity_rate_noise=connectivity_rate_noise,
                                      symmetric_flag=symmetric_flag,
                                      random=random)

    print(dataset)


if __name__ == "__main__":
    tst_generate_graphs_dataset()
