import copy
import numpy as np


class GraphSample:

    def __init__(self,
                 num_nodes,
                 nodes_vecs,
                 edges_full):
        self.num_nodes = num_nodes
        self.nodes_vecs = nodes_vecs
        self.edges_full = edges_full
        self.edges_list = None

    def remove_nodes(self, nodes_list):
        np.delete(self.edges_full, nodes_list, axis=0)
        np.delete(self.edges_full, nodes_list, axis=1)
        np.delete(self.nodes_vecs, nodes_list, axis=0)
        self.num_nodes -= len(nodes_list)

    def scramble(self, idx):
        self.edges_full = self.edges_full[idx, :]
        self.edges_full = self.edges_full[:, idx]
        self.nodes_vecs = self.nodes_vecs[idx, :]

    def get_edges_list(self):
        if self.edges_list is None:
            self.edges_list = list()
            for y in range(self.num_nodes):
                for x in range(self.num_nodes):
                    if self.edges_full[y, x] != 0:
                        self.edges_list.append([x, y])
            self.edges_list = np.array(self.edges_list).T
        return self.edges_list

    def __str__(self):
        s = list()
        for y in range(self.num_nodes):
            for x in range(self.num_nodes):
                s.append("0" if self.edges_full[y][x]==0 else "1")
            s.append("\t")
            s.append(f"{' '.join([f'{num:+2.4f}' for num in self.nodes_vecs[y]])}")
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
            s.append(f"center {c}\n")
            s.append(f"{self.centers[c].__str__()}\n")
        s.append(f"samples=\n")
        for i in range(len(self.samples)):
            s.append(f"sample {i} label={self.labels[i]}\n")
            s.append(f"{self.samples[i].__str__()}\n")
        return "".join(s)


def create_edges(num_nodes, connectivity_rate, symmetric_flag, random):
    edges_full = np.zeros(shape=(num_nodes * num_nodes, 1))
    edges_full[:int(num_nodes * num_nodes * connectivity_rate)] = 1
    random.shuffle(edges_full)
    edges_full = edges_full.reshape((num_nodes, num_nodes))
    if symmetric_flag:
        edges_full = add_edges(edges_full, edges_full.T)
    return edges_full


def add_edges(edges1, edges2):
    edges = edges1 + edges2
    edges[edges > 1] = 1
    return edges


def create_centers(num_classes, min_nodes, max_nodes, connectivity_rate, symmetric_flag, dim_nodes, random):
    centers = list()

    for c in range(num_classes):
        # decide num of nodes
        num_nodes = random.randint(min_nodes, max_nodes+1)

        edges_full = np.zeros(shape=(num_nodes * num_nodes, 1))
        edges_full[:int(num_nodes * num_nodes * connectivity_rate)] = 1
        random.shuffle(edges_full)
        edges_full = edges_full.reshape((num_nodes, num_nodes))
        if symmetric_flag:
            edges_full = edges_full + edges_full.T
            edges_full[edges_full > 1] = 1
        vector = random.normal(size=(num_nodes, dim_nodes))
        centers.append(GraphSample(num_nodes=num_nodes, nodes_vecs=vector, edges_full=edges_full))
    return centers


def noise_centers(graph_sample, noise_nodes, random):
    graph_sample.nodes_vecs = graph_sample.nodes_vecs +  random.normal(size=graph_sample.nodes_vecs.shape) * noise_nodes


def noise_connectivity(graph_sample, connectivity_rate_noise, symmetric_flag, random):
    edges_full_noise = create_edges(graph_sample.num_nodes, connectivity_rate_noise, symmetric_flag, random)
    noisy_edges = graph_sample.edges_full + edges_full_noise
    noisy_edges[noisy_edges == 2] = 0  # Make flip of edges
    graph_sample.edges_full = noisy_edges


def remove_node_noise(graph_sample, prob_remove, random):
    nodes_to_remove = random.choice(2, size=(graph_sample.num_nodes), p=[1 - prob_remove, prob_remove])
    idx_to_remove = np.where(nodes_to_remove == 1)[0].tolist()
    graph_sample.remove_nodes(idx_to_remove)


def scramble_graph(graph_sample, random):
    idx = list(range(graph_sample.num_nodes))
    idx = random.permutation(idx)
    graph_sample.scramble(idx)


def generate_graphs_dataset(num_samples,
                            num_classes,
                            min_nodes,
                            max_nodes,
                            dim_nodes,
                            noise_nodes,
                            connectivity_rate,
                            connectivity_rate_noise,
                            noise_remove_node,
                            noise_add_node,
                            symmetric_flag,
                            random):
    # Generate classes centers
    centers = create_centers(num_classes, min_nodes, max_nodes, connectivity_rate, symmetric_flag, dim_nodes, random)
    samples = list()
    labels = list()

    # Create samples and labels
    for i in range(num_samples):
        # get a class
        c = random.randint(num_classes)
        labels.append(c)

        # Get the sample
        sample = copy.deepcopy(centers[c])
        # print(f"original=\n{center}")

        # noise the vector nodes
        noise_centers(sample, noise_nodes, random)
        # print(f"noise_centers=\n{center}")

        # noise the edges_full
        noise_connectivity(sample, connectivity_rate_noise, symmetric_flag, random)
        # print(f"noise_connectivity=\n{center}")

        # Remove edge noise
        remove_node_noise(sample, noise_remove_node, random)
        # print(f"remove_node_noise=\n{center}")

        # Scramble
        scramble_graph(sample, random)
        # print(f"scramble_graph=\n{center}")

        samples.append(sample)

    # Generate samples
    return GraphDataset(samples=samples, labels=labels, centers=centers)

