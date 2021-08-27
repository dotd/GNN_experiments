import time

import numpy as np
import pickle
import torch
import gc
import copy

from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

import tst.ogb.exp_utils
from definitions_gnn_experiments import ROOT_DIR
from src.utils.proxy_utils import set_proxy
from src.utils.file_utils import create_folder_safe, save_str, file_exists

# Download dataset


def load_TU_dataset(name = 'MUTAG'):
    pickle_path = f'{ROOT_DIR}/data_fast/TUDataset/{name}/'
    pickle_file = f"{name}.p"
    pickle_full = f"{pickle_path}/{pickle_file}"
    if file_exists(pickle_full):
        print("path exists - load fast pickle")
        f = open(pickle_full, "rb")
        gc.disable()
        dataset = pickle.load(f)
        gc.enable()
        f.close()
    else:
        print("Did not load yet: download data and store")
        from torch_geometric.datasets import TUDataset
        set_proxy()
        dataset = TUDataset(root=f'{ROOT_DIR}/data/TUDataset', name=name)
        create_folder_safe(pickle_path)
        pickle.dump(dataset, open(pickle_full, "wb"), protocol=-1)
    return dataset


def show_dataset_stats(dataset):
    # Show the dataset stats
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Show the first graph.
    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def split_to_train_test(dataset):
    torch.manual_seed(12345)
    # dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    """
    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
    """
    return train_dataset, test_dataset, train_loader, test_loader


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels, in_size, out_size, conv_ctr=GCNConv):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = conv_ctr(in_size, hidden_channels)
        self.conv2 = conv_ctr(hidden_channels, hidden_channels)
        self.conv3 = conv_ctr(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_size)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT(torch.nn.Module):

    def __init__(self, dim_nodes, num_classes, heads, num_hidden):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_nodes, num_hidden, heads=heads)
        self.conv2 = GATConv(num_hidden*heads, num_hidden, heads=heads)
        self.conv3 = GATConv(num_hidden*heads, num_hidden, heads=heads, concat=False)
        self.lin = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train(args, model, train_loader, lr=0.01, log_every=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    for batch_i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if batch_i % log_every == 0:
            print(f"Batch {batch_i + 1} / {len(train_loader)} | {(time.time() - start_time) / (batch_i + 1)} sec / iteration")

    return (time.time() - start_time) / len(train_loader)


def func_test(args, model, loader, log_every=50):
    model.eval()

    correct = 0
    start_time = time.time()
    for batch_i, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        if batch_i % log_every == 0:
            print(f"Batch {batch_i + 1} / {len(loader)} | {(time.time() - start_time) / (batch_i + 1)} sec / iteration")
    return correct / len(loader.dataset), (time.time() - start_time) / len(loader)


def sparsify_using_lsh():
    pass


def sparsify_randomly(dataset, sparsification_rate, random):
    for i in range(len(dataset)):
        length = dataset[i].edge_index.shape[1]
        length_final = int(length * sparsification_rate)
        indices = list(range(length))
        indices = random.permutation(indices)
        indices = indices[:length_final]
        dataset[i].edge_index = dataset[i].edge_index[:, indices]
        dataset[i].edge_attr = dataset[i].edge_attr[indices, :]
    return dataset


def classify(dataset, model_in_size, num_classes):
    # Show some stats
    # show_dataset_stats(dataset)
    # Split to train and test
    train_dataset, test_dataset, train_loader, test_loader = split_to_train_test(dataset)
    model = GCN(hidden_channels=64, in_size=model_in_size, out_size=num_classes)
    print(model)

    test_acc = func_test(model, test_loader)
    print(f'Test Acc: {test_acc:.4f}')

    for epoch in range(10):
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def run():
    random = np.random.RandomState(0)
    # Load the dataset
    dataset_orig = load_TU_dataset()
    num_classes = dataset_orig.num_classes
    in_size = dataset_orig.num_node_features
    dataset_orig = list(dataset_orig)
    dataset_sparse = copy.deepcopy(dataset_orig)
    dataset_sparse = sparsify_randomly(dataset_sparse, 0.1, random=random)
    print("Classify dataset_orig")
    classify(dataset_orig, model_in_size=in_size, num_classes=num_classes)
    print("Classify dataset_sparse")
    classify(dataset_sparse, model_in_size=in_size, num_classes=num_classes)


if __name__ == "__main__":
    run()
