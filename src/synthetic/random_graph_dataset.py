import numpy as np


class GraphSample:

    def __init__(self,
                 num_nodes,
                 edges,
                 nodes_vecs,
                 edges_full):
        self.num_nodes = num_nodes
        self.edges = edges
        self.nodes_vecs = nodes_vecs
        self.edges_full = edges_full

    def __str__(self):
        s = list()
        for y in range(self.num_nodes):
            for x in range(self.num_nodes):
                s.append("0" if self.edges_full[y][x]==0 else "1")
            s.append("\t")
            s.append(f"{self.nodes_vecs[y]}")
            s.append("\n")
        return "".join(s)


class GraphDataset:

    def __init__(self,
                 samples=None,
                 labels=None,
                 centers=None):
        self.samples = samples
        self.labels = labels
        self.centers = centers

    def __str__(self):
        s = list()
        s.append(f"centers=\n")
        for c in range(len(self.centers)):
            s.append(f"{self.centers[c].__str__()}\n")
        return "".join(s)


def generate_graphs_dataset(num_samples,
                            num_classes,
                            min_nodes,
                            max_nodes,
                            dim_nodes,
                            class_nodes_var,
                            noise_nodes,
                            connectivity_rate,
                            connectivity_rate_noise,
                            symmetric_flag,
                            random):
    # Generate classes centers
    samples = list()
    labels = list()
    centers = list()

    connectivity_mats = list()
    nodes_vectors = list()
    for c in range(num_classes):
        # decide num of nodes
        num_nodes = random.randint(min_nodes, max_nodes)
        connectivity_mat = np.zeros(shape=(num_nodes * num_nodes, 1))
        connectivity_mat[:int(num_nodes * num_nodes * connectivity_rate)] = 1
        random.shuffle(connectivity_mat)
        connectivity_mat = connectivity_mat.reshape((num_nodes, num_nodes))
        print(connectivity_mat)
        vector = random.normal(size=(num_nodes, dim_nodes))

        connectivity_mats.append(connectivity_mat)
        nodes_vectors.append(vector)
        centers.append(GraphSample(num_nodes=num_nodes, edges=None, nodes_vecs=vector, edges_full=connectivity_mat))


    # Generate samples
    return GraphDataset(samples=samples, labels=labels, centers=centers)

