import copy
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

"""
Convetion:

Matrix [y][x]
list [y,x]
"""


class GraphSample:

    def __init__(self,
                 num_nodes,
                 nodes_vecs,
                 edges_list,
                 edges_vecs):
        self.num_nodes = num_nodes
        self.nodes_vecs = nodes_vecs
        self.edges_list = edges_list
        self.edges_vecs = edges_vecs
        self.edges_full = None
        self.num_edges = len(self.edges_list[0])

    def remove_nodes(self, nodes_list):
        old_node_names = [idx for idx in np.arange(self.num_nodes) if idx not in nodes_list]
        self.edges_full = np.delete(self.edges_full, nodes_list, axis=0)
        self.edges_full = np.delete(self.edges_full, nodes_list, axis=1)
        self.nodes_vecs = np.delete(self.nodes_vecs, nodes_list, axis=0)

        columns_to_remove = np.array([])
        self.num_nodes -= len(nodes_list)
        new_node_names = [idx for idx in np.arange(self.num_nodes)]

        for node in nodes_list:
            node_idx_from, = np.where(self.edges_list[0] == node)
            node_idx_to, = np.where(self.edges_list[1] == node)
            columns_to_remove = np.concatenate([columns_to_remove, node_idx_from, node_idx_to])

        self.edges_vecs = np.delete(self.edges_vecs, columns_to_remove.astype(int), axis=0)
        self.edges_list = np.delete(self.edges_list, columns_to_remove.astype(int), axis=1)
        self.edges_list = self.rename_nodes_in_edge_list(old_node_names, new_node_names, self.edges_list)

    def rename_nodes_in_edge_list(self, sources, targets, edge_list):
        new_edges_list = edge_list.copy()

        for source, target in zip(sources, targets):
            # the value stored in idx is the old index
            new_edges_list[self.edges_list == source] = target

        return new_edges_list

    def scramble(self, idx):
        self.edges_full = self.edges_full[idx, :]
        self.edges_full = self.edges_full[:, idx]
        self.nodes_vecs = self.nodes_vecs[idx, :]

        # need to create a copy in order to not override the new changes
        new_edges_list = self.edges_list.copy()

        self.edges_list = self.rename_nodes_in_edge_list(sources=idx, targets=np.arange(len(idx)), edge_list=self.edges_list)

    def get_edges_full(self):
        if self.edges_full is None:
            self.edges_full = np.zeros(shape=(self.num_nodes, self.num_nodes))
            for i in range(self.num_edges):
                self.edges_full[self.edges_list[0][i], self.edges_list[1][i]] = 1
        return self.edges_full

    def get_edges_list(self):
        if self.edges_list is None:
            self.edges_list = list()
            for y in range(self.num_nodes):
                for x in range(self.num_nodes):
                    if self.edges_full[y][x] != 0:
                        self.edges_list.append([y, x])
            self.edges_list = np.array(self.edges_list).T
        return self.edges_list

    def __str__(self):
        s = list()
        for y in range(self.num_nodes):
            for x in range(self.num_nodes):
                s.append("0" if self.get_edges_full()[y][x] == 0 else "1")
            s.append("\t")
            s.append(f"{' '.join([f'{num:+2.4f}' for num in self.nodes_vecs[y]])}")
            s.append("\n")

        for e in range(self.num_edges):
            s.append(f"{self.edges_list[0][e]}->{self.edges_list[1][e]}")
            s.append("\t")
            s.append(f"{' '.join([f'{num:+2.4f}' for num in self.edges_vecs[e]])}")
            s.append("\n")
        return "".join(s)


def graph_sample_to_networkx(graph_sample):
    graph_nx = nx.Graph()
    # Transform nodes
    for idx in range(graph_sample.nodes_vecs.shape[0]):
        vec = graph_sample.nodes_vecs[idx].tolist()
        graph_nx.add_nodes_from([(idx, {"x": vec})])
    # Transforms edges
    for edge in graph_sample.get_edges_list().T:
        graph_nx.add_edge(edge[0], edge[1])
    return graph_nx


def graph_sample_dataset_to_networkx(graph_sample_dataset):
    print("Transforming graph samples samples to networkx")
    samples_networkx = list()
    centers_networkx = list()
    # Transform the samples
    for sample in tqdm(graph_sample_dataset.samples):
        sample_networkx = graph_sample_to_networkx(sample)
        samples_networkx.append(sample_networkx)
    # Transform the centers
    for center in graph_sample_dataset.centers:
        center_networkx = graph_sample_to_networkx(center)
        centers_networkx.append(center_networkx)
    graph_sample_dataset.samples = samples_networkx
    graph_sample_dataset.centers = centers_networkx


class GraphSampleDataset:

    def __init__(self,
                 samples=None,
                 labels=None,
                 centers=None):
        self.samples = samples
        self.labels = labels
        self.centers = centers
        for sample in samples:
            sample.get_edges_list()

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


def edges_mat_to_list(mat):
    N = mat.shape[0]
    lst = list()
    for y in range(N):
        for x in range(N):
            if mat[y][x] != 0:
                lst.append([y, x])
    return np.array(lst).T


def create_centers(num_classes,
                   min_nodes,
                   max_nodes,
                   connectivity_rate,
                   symmetric_flag,
                   dim_nodes,
                   dim_edges,
                   centers_nodes_std,
                   centers_edges_std,
                   random):
    centers = list()

    for c in range(num_classes):
        # decide num of nodes
        num_nodes = random.randint(min_nodes, max_nodes + 1)

        edges_full = np.zeros(shape=(num_nodes * num_nodes, 1))
        edges_full[:int(num_nodes * num_nodes * connectivity_rate)] = 1
        random.shuffle(edges_full)
        edges_full = edges_full.reshape((num_nodes, num_nodes))
        if symmetric_flag:
            edges_full = edges_full + edges_full.T
            edges_full[edges_full > 1] = 1
        # Get edges list
        edges_list = edges_mat_to_list(edges_full)

        num_edges = np.count_nonzero(edges_full)
        nodes_vecs = random.normal(size=(num_nodes, dim_nodes)) * centers_nodes_std
        edges_vecs = random.normal(size=(num_edges, dim_edges)) * centers_edges_std
        centers.append(GraphSample(num_nodes=num_nodes,
                                   nodes_vecs=nodes_vecs,
                                   edges_list=edges_list,
                                   edges_vecs=edges_vecs))
    return centers


def noise_the_nodes(graph_sample, node_additive_noise_std, random):
    graph_sample.nodes_vecs = graph_sample.nodes_vecs + \
                              random.normal(size=graph_sample.nodes_vecs.shape) * node_additive_noise_std


def noise_the_edges(graph_sample, edge_additive_noise_std, random):
    graph_sample.edges_vecs = graph_sample.edges_vecs + \
                              random.normal(size=graph_sample.edges_vecs.shape) * edge_additive_noise_std


def noise_connectivity(graph_sample, connectivity_rate_noise, symmetric_flag, random):
    edges_full_noise = create_edges(graph_sample.num_nodes, connectivity_rate_noise, symmetric_flag, random)
    noisy_edges = graph_sample.get_edges_full() + edges_full_noise
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


# def add_nodes(sample):
#     nodes_vecs = random.normal(size=(num_nodes, dim_nodes)) * centers_nodes_std


def generate_graphs_dataset(num_samples,
                            num_classes,
                            min_nodes,
                            max_nodes,
                            dim_nodes,
                            dim_edges,
                            connectivity_rate,
                            connectivity_rate_noise,
                            noise_remove_node,
                            symmetric_flag,
                            centers_nodes_std,
                            centers_edges_std,
                            nodes_order_scramble_flag,
                            node_additive_noise_std,
                            edge_additive_noise_std,
                            noise_add_node,
                            random,
                            **kwargs):

    # Generate classes centers
    centers = create_centers(num_classes,
                             min_nodes,
                             max_nodes,
                             connectivity_rate,
                             symmetric_flag,
                             dim_nodes=dim_nodes,
                             dim_edges=dim_edges,
                             centers_nodes_std=centers_nodes_std,
                             centers_edges_std=centers_edges_std,
                             random=random)
    samples = list()
    labels = np.random.randint(num_classes, size=num_samples)

    # Create samples and labels
    for i in range(num_samples):
        # get a class
        c = labels[i]

        # Get the sample
        sample = copy.deepcopy(centers[c])
        # print(f"original=\n{center}")

        # noise the vector nodes
        noise_the_nodes(sample, node_additive_noise_std, random)
        noise_the_edges(sample, edge_additive_noise_std, random)
        # print(f"noise_centers=\n{center}")

        # noise the edges_full
        noise_connectivity(sample, connectivity_rate_noise, symmetric_flag, random)
        # print(f"noise_connectivity=\n{center}")

        # Remove edge noise
        remove_node_noise(sample, noise_remove_node, random)
        # print(f"remove_node_noise=\n{center}")

        # Scramble
        if nodes_order_scramble_flag:
            scramble_graph(sample, random)
        # print(f"scramble_graph=\n{center}")

        # add random nodes
        # add_nodes(sample)

        samples.append(sample)

    # Generate samples
    gsd = GraphSampleDataset(samples=samples, labels=labels, centers=centers)
    return gsd


def add_random_gaussian_edge_attr(tg_dataset, dim_features, random):
    for tg_sample in tg_dataset:
        num_edges = tg_sample.edge_index.shape[1]
        tg_sample.edge_attr = torch.FloatTensor(random.normal(size=(num_edges, dim_features)))
