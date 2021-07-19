import time

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import Reddit, Amazon, Planetoid
from tqdm import tqdm

from src.archs.gat_sage import GATSage
from src.archs.node_prediction import NodeGat, NodeARMA
from src.archs.sage import SAGE
from src.utils.csv_utils import prepare_csv
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.ogb.main_pyg_with_pruning import prune_dataset, get_args

start_time = time.time()
# Created by: Eitan Kosman, BCAI

sage = False


def train(epoch, dataset, train_loader, model, device, optimizer, tb_writer):
    """
    Performs a training episode.

    @param epoch: the epoch number
    @param dataset: a pytorch geometric dataset object
    @param train_loader: dataset sampler for the node prediction task
    @param model: GNN to use for inference
    @param device: the device on which we perform the calculations
    @param optimizer: a pytorch optimizer object for updating the parameters of the GNN
    """
    global sage

    model.train()

    pbar = tqdm(total=int(dataset.data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    x = dataset.data.x.to(device)
    y = dataset.data.y.squeeze().to(device)
    y_pred = torch.tensor([])
    y_true = torch.tensor([])
    start_time = time.time()

    if sage:
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
            y_pred = torch.cat([y_pred, out.argmax(dim=-1).detach().cpu()])
            y_true = torch.cat([y_true, y[n_id[:batch_size]].detach().cpu()])

            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_iterations', loss.item(), tb_writer.iteration)
                tb_writer.iteration += 1
    else:
        optimizer.zero_grad()
        log_logits = model(dataset.data.x.to(device), dataset.data.edge_index.to(device))
        loss = F.nll_loss(log_logits[dataset.data.train_mask], dataset.data.y[dataset.data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        total_correct += int(log_logits.argmax(dim=-1)[dataset.data.train_mask].detach().cpu().eq(dataset.data.y[dataset.data.train_mask]).sum())
        total_loss = loss.item()
        y_pred = log_logits.argmax(dim=-1)[dataset.data.train_mask].detach().cpu()
        y_true = dataset.data.y[dataset.data.train_mask].detach().cpu()

    pbar.close()

    approx_acc = total_correct / int(dataset.data.train_mask.sum())
    f1 = f1_score(y_true, y_pred, average='micro')

    if sage:
        avg_time = (time.time() - start_time) / len(train_loader.dataset)
        loss = total_loss / len(train_loader)
    else:
        avg_time = time.time() - start_time
        loss = total_loss / len(dataset.data.y)

    return loss, approx_acc, f1, avg_time


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
    start_time = time.time()

    if sage:
        out = model.inference(dataset.data.x.to(device), subgraph_loader, device)
    else:
        out = model(dataset.data.x.to(device), dataset.data.edge_index.to(device))

    y_true = dataset.data.y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    results = []
    for mask in [dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask]:
        results += [accuracy_score(y_true[mask], y_pred[mask])]

    return (*results, (time.time() - start_time) / dataset.data.x.shape[0])


def get_dataset(dataset_name):
    """
    Retrieves the dataset corresponding to the given name.
    """
    print("Getting dataset...")
    path = 'dataset'
    if dataset_name == 'reddit':
        dataset = Reddit(path)
    elif dataset_name in ['amazon_comp', 'amazon_photo']:
        dataset = Amazon(path, "Computers", T.NormalizeFeatures()) if dataset_name == 'amazon_comp' else Amazon(path, "Photo", T.NormalizeFeatures())
        data = dataset.data
        idx_train, idx_test = train_test_split(list(range(data.x.shape[0])), test_size=0.4, random_state=42)
        idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)
        data.train_mask = torch.tensor(idx_train)
        data.val_mask = torch.tensor(idx_val)
        data.test_mask = torch.tensor(idx_test)
        dataset.data = data
    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name=dataset_name, split="full", transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError

    print("Dataset ready!")
    return dataset


def get_model(num_features, num_classes, arch):
    global sage
    """
    Retrieves the model corresponding to the given name.

    @param num_features: the dimensionality of the node features
    @param num_classes: number of target labels
    @param arch: name of the GNN architecture
    """
    if arch == 'sage':
        model = SAGE(in_channels=num_features, out_channels=num_classes)
        sage = True
    elif arch == 'gat_sage':
        model = GATSage(num_features=num_features, num_classes=num_classes)
        sage = True
    elif arch == 'gat':
        model = NodeGat(num_features=num_features, num_classes=num_classes, num_hidden=8, num_heads=2)
        sage = False
    elif arch == 'arma':
        model = NodeARMA(num_features=num_features, num_classes=num_classes)
        sage = False
    else:
        raise NotImplementedError

    return model


def run(args):
    dataset = get_dataset(args.dataset)
    data = dataset.data
    tb_writer = SummaryWriter()
    tb_writer.iteration = 0

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and args.device != 'cpu' else torch.device("cpu")
    model = get_model(dataset.data.num_features, dataset.num_classes, args.gnn)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    old_edge_count = data.edge_index.shape[1]

    # Pass the whole graph to the pruning mechanism. Consider it as one sample
    pruning_params, prunning_ratio = prune_dataset([data], args, random=np.random.RandomState(0), pruning_params=None)

    edge_count = data.edge_index.shape[1]
    print(
        f"Old number of edges: {old_edge_count}. New one: {edge_count}. Change: {(old_edge_count - edge_count) / old_edge_count * 100}\%")

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=1024, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)

    best_train = 0
    best_test = 0
    train_times = []
    test_times = []
    for epoch in range(1, args.epochs + 1):
        loss, acc, f1, avg_time_train = train(epoch, dataset, train_loader, model, device, optimizer, tb_writer)
        train_times.append(avg_time_train)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {f1:.4f}')

        train_acc, val_acc, test_acc, avg_time_test = test(dataset, subgraph_loader, model, device)
        test_times.append(avg_time_test)
        print(f'Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}, '
              f'Test ACC: {test_acc:.4f}')

        tb_writer.add_scalars('Accuracy',
                              {'train': train_acc,
                               'Validation': val_acc,
                               'Test': test_acc},
                              epoch)

        best_train = max(best_train, train_acc)
        best_test = max(best_test, test_acc)

    return prunning_ratio, best_train, best_test, avg_time_train, avg_time_test


@prepare_csv
def main(args, csv_file):
    vals = dict()

    """
    Pruning with LSH
    """
    args.pruning_method = 'minhash_lsh'
    tb_writer = None
    if args.enable_clearml_logger:
        tb_writer = SummaryWriter(log_dir=None)
        tags = [
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
        ]
        pruning_param_name = 'num_minhash_funcs' if args.pruning_method == 'minhash_lsh' else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if args.pruning_method == 'minhash_lsh' else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name=f"GNN_{args.dataset}_{args.gnn}_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = run(args)
    print(f"{time.time() - start_time:.4f} end time")

    vals['keep edges'] = prunning_ratio
    vals['minhash train'] = best_train
    vals['minhash test'] = best_test

    vals['minhash time train'] = avg_time_train
    vals['minhash time test'] = avg_time_test

    print("Sleeping between experiments...")
    time.sleep(5)
    print("Wake up!")
    """
    Pruning with random
    """
    args.pruning_method = 'random'
    args.random_pruning_prob = prunning_ratio
    tb_writer = None
    if args.enable_clearml_logger:
        clearml_logger.close()
        tb_writer = SummaryWriter(log_dir=None)
        tags = [
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
        ]
        pruning_param_name = 'num_minhash_funcs' if args.pruning_method == 'minhash_lsh' else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if args.pruning_method == 'minhash_lsh' else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name=f"GNN_{args.dataset}_{args.gnn}_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = run(args)
    print(f"{time.time() - start_time:.4f} end time")
    vals['random train'] = best_train
    vals['random test'] = best_test
    vals['random time train'] = avg_time_train
    vals['random time test'] = avg_time_test
    df = pd.read_csv(csv_file)
    df = df.append(vals, ignore_index=True)
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    args = get_args()
    args.csv_file = f"{args.dataset}_{args.gnn}_results.csv"
    main(args)
