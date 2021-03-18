import numpy as np
from src.synthetic.random_graph_dataset import GraphSample


def tst_GraphSample():
    rnd = np.random.RandomState(0)
    num_nodes = 5
    nodes_vecs = np.array([[1,2], [3,4], [5,6], [7,8], [9,0]])
    edges_list = [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 3]]
    edges_vecs = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    graph_sample = GraphSample(num_nodes=num_nodes, nodes_vecs=nodes_vecs, edges_list=edges_list, edges_vecs=edges_vecs)
    print(graph_sample.__str__())


if __name__ == '__main__':
    tst_GraphSample()
