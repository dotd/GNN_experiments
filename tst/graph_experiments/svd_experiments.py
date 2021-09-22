import copy
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.synthetic.random_graph_dataset import create_centers, noise_connectivity


def IKSVD(A, k=5):
    u, s, vh = np.linalg.svd(A)
    A_ = u[:, :k] * s[:k] @ vh[:k, :]
    A_ = np.round(A_)
    return A_


if __name__ == '__main__':
    centers = create_centers(num_classes=100,
                             min_nodes=100,
                             max_nodes=100,
                             connectivity_rate=0.1,
                             symmetric_flag=True,
                             dim_nodes=10,
                             dim_edges=10,
                             centers_nodes_std=1,
                             centers_edges_std=1,
                             random=np.random.RandomState(0))

    num_experiments_per_graph = 100
    k = 3

    bar = tqdm(total=len(centers) * num_experiments_per_graph)
    points = []
    for M in centers:
        A = M.get_edges_full()
        A_sparse = IKSVD(A, k=k)
        # A_minus_A_sparse_norm = np.linalg.norm(A - A_sparse, ord='fro')
        for i in range(num_experiments_per_graph):
            M_tag = copy.deepcopy(M)
            noise_connectivity(graph_sample=M_tag, connectivity_rate_noise=random.random(), symmetric_flag=True, random=np.random.RandomState(0))
            A_tag = M_tag.get_edges_full()
            A_tag_sparse = IKSVD(A_tag, k=k)
            # A_tag_minus_A_sparse_norm = np.linalg.norm(A_tag - A_tag_sparse, ord='fro')

            A_minus_A_tag = np.linalg.norm(A - A_tag, ord='fro')
            A_sparse_minus_A_tag_sparse = np.linalg.norm(A_sparse - A_tag_sparse, ord='fro')
            points.append((A_minus_A_tag, A_sparse_minus_A_tag_sparse))
            bar.update(1)

    plt.figure()
    x = [e[0] for e in points]
    y = [e[1] for e in points]
    plt.scatter(x, y)
    plt.xlabel('|M-M\'|_F')
    plt.ylabel('|Sparse(M)-Sparse(M\')_F')
    plt.show()
