import time
start_time = time.time()

from torch_geometric.data import DataLoader # noqa
from torch_geometric.datasets import GNNBenchmarkDataset # noqa

from tst.torch_geometric.tst_torch_geometric1 import GCN # noqa
from tst.torch_geometric.tst_torch_geometric1 import train, func_test # noqa
from src.utils.proxy_utils import set_proxy


def tst_classify_synthetic():
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")
    dataset_name = "MNIST"
    set_proxy()
    train_dataset = GNNBenchmarkDataset(root="tst/gnn_benchmark_datasets", name=dataset_name)
    test_dataset = GNNBenchmarkDataset(root="tst/gnn_benchmark_datasets", name=dataset_name, split="test")
    dim_nodes = train_dataset.data.x.shape[1]
    num_classes = train_dataset.num_classes

    print(f"{time.time() - start_time:.4f} Finished Loading the dataset: {dataset_name}")
    print(f"Number of classes: {num_classes}. Node feature shape: {dim_nodes}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = GCN(hidden_channels=60, in_size=dim_nodes, out_size=num_classes)

    test_acc = func_test(model, test_loader)
    print(f'{time.time() - start_time:.2f}s Test Acc (Initial): {test_acc:.4f}')

    for epoch in range(10):
        train(model, train_loader)
        train_acc = func_test(model, train_loader)
        test_acc = func_test(model, test_loader)
        print(f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    print(f"{time.time() - start_time:.4f} start time")
    tst_classify_synthetic()
