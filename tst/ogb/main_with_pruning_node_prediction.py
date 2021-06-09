import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import Reddit, Amazon, Planetoid
from tqdm import tqdm

from src.archs.gat_sage import GATSage
from src.archs.mlp_node_prediction import MLP
from src.archs.sage import SAGE
from tst.ogb.main_pyg_with_pruning import prune_dataset, get_args


def train(epoch, dataset, train_loader, model, device, optimizer):
    """
    Performs a training episode.

    @param epoch: the epoch number
    @param dataset: a pytorch geometric dataset object
    @param train_loader: dataset sampler for the node prediction task
    @param model: GNN to use for inference
    @param device: the device on which we perform the calculations
    @param optimizer: a pytorch optimizer object for updating the parameters of the GNN
    """

    model.train()

    pbar = tqdm(total=int(dataset.data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    x = dataset.data.x.to(device)
    y = dataset.data.y.squeeze().to(device)
    y_pred = torch.tensor([])
    y_true = torch.tensor([])

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)
        y_pred = torch.cat([y_pred, out.argmax(dim=-1)])
        y_true = torch.cat([y_true, y[n_id[:batch_size]]])

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(dataset.data.train_mask.sum())
    f1 = f1_score(y_true, y_pred, average='micro')

    return loss, approx_acc, f1


@torch.no_grad()
def test(dataset, subgraph_loader, model, device):
    """
    Performs a testing episode.

    @param dataset: a pytorch geometric dataset object
    @param subgraph_loader: dataset sampler for the node prediction task
    @param model: GNN to use for inference
    @param device: the device on which we perform the calculations
    """

    model.eval()

    out = model.inference(dataset.data.x.to(device), subgraph_loader, device)

    y_true = dataset.data.y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask]:
        results += [f1_score(y_true[mask], y_pred[mask], average='micro')]

    return results


def get_dataset(dataset_name):
    """
    Retrieves the dataset corresponding to the given name.
    """
    path = 'dataset'
    if dataset_name == 'reddit':
        dataset = Reddit(path)
    elif dataset_name == 'amazon_comp':
        dataset = Amazon(path, name="Computers")
        data = dataset.data
        idx_train, idx_test = train_test_split(list(range(data.x.shape[0])), test_size=0.4, random_state=42)
        idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)
        data.train_mask = torch.tensor(idx_train)
        data.val_mask = torch.tensor(idx_val)
        data.test_mask = torch.tensor(idx_test)
        dataset.data = data
    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name=dataset_name, split="full", )
    else:
        raise NotImplementedError

    return dataset


def get_model(num_features, num_classes, arch):
    """
    Retrieves the model corresponding to the given name.

    @param num_features: the dimensionality of the node features
    @param num_classes: number of target labels
    @param arch: name of the GNN architecture
    """
    if arch == 'sage':
        model = SAGE(num_features, 256, num_classes)
    elif arch == 'gat':
        model = GATSage(num_features, num_classes)
    elif arch == 'mlp':
        model = MLP(num_features, num_classes, 2)
    else:
        raise NotImplementedError

    return model


def main():
    args = get_args()
    dataset = get_dataset(args.dataset)
    data = dataset.data

    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(dataset.data.num_features, dataset.num_classes, args.gnn)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    old_edge_count = data.edge_index.shape[1]

    # Pass the whole graph to the pruning mechanism. Consider it as one sample
    # prune_dataset([data], args, random=np.random.RandomState(0), pruning_params=None)

    edge_count = data.edge_index.shape[1]
    print(
        f"Old number of edges: {old_edge_count}. New one: {edge_count}. Change: {(old_edge_count - edge_count) / old_edge_count * 100}\%")

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=1024, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)

    for epoch in range(1, args.epochs + 1):
        loss, acc, f1 = train(epoch, dataset, train_loader, model, device, optimizer)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {f1:.4f}')

        train_f1, val_f1, test_f1 = test(dataset, subgraph_loader, model, device)
        print(f'Train: {train_f1:.4f}, Val: {val_f1:.4f}, '
              f'Test: {test_f1:.4f}')


if __name__ == '__main__':
    main()
