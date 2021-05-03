import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx


def convert_to_line_graphs(dataset, directed=False):
    ctor = nx.DiGraph if directed else nx.Graph

    for sample in dataset:
        G = to_networkx(sample,
                        node_attrs=['x'],
                        edge_attrs=['edge_attr'],
                        to_undirected=not directed)
        line_graph = nx.line_graph(G, create_using=ctor)
        geom_sample = from_networkx(line_graph)
        i=1

    pass