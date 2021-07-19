import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Reddit, Amazon, Planetoid
from tqdm import tqdm

from src.archs.head import MLP
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.ogb.main_pyg_with_pruning import prune_dataset, get_args


# Created by: Eitan Kosman, BCAI


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

    model.train()

    pbar = tqdm(total=int(dataset.data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    y_pred = torch.tensor([])
    y_true = torch.tensor([])

    for x, y in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(x.shape[0])
        y_pred = torch.cat([y_pred, out.argmax(dim=-1).detach().cpu()])
        y_true = torch.cat([y_true, y.detach().cpu()])

        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train_iterations', loss.item(), tb_writer.iteration)
            tb_writer.iteration += 1

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(dataset.data.train_mask.sum())
    f1 = f1_score(y_true, y_pred, average='micro')

    return loss, approx_acc, f1


@torch.no_grad()
def test(dataset, model, device):
    """
    Performs a testing episode.

    @param dataset: a pytorch geometric dataset object
    @param subgraph_loader: dataset sampler for the node prediction task
    @param model: GNN to use for inference
    @param device: the device on which we perform the calculations
    """

    model.eval()

    out = model(dataset.data.x.to(device))

    y_true = dataset.data.y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()

    results = []
    for mask in [dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask]:
        results += [accuracy_score(y_true[mask], y_pred[mask])]

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

        train_mask = torch.tensor([False] * data.x.shape[0])
        val_mask = torch.tensor([False] * data.x.shape[0])
        test_mask = torch.tensor([False] * data.x.shape[0])

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        dataset.data = data
    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name=dataset_name, split="full", )
    else:
        raise NotImplementedError

    return dataset


def get_model(num_features, num_classes):
    """
    Retrieves the model corresponding to the given name.

    @param num_features: the dimensionality of the node features
    @param num_classes: number of target labels
    """
    return MLP(in_features=num_features, out_features=num_classes)


def main():
    args = get_args()
    dataset = get_dataset(args.dataset)
    data = dataset.data
    tb_writer = SummaryWriter()
    tb_writer.iteration = 0

    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() and args.device != 'cpu' else torch.device("cpu")
    model = get_model(dataset.data.num_features, dataset.num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader([(x_, y_) for i, (x_, y_) in enumerate(zip(data.x, data.y)) if data.train_mask[i]],
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True, )

    if args.enable_clearml_logger:
        tags = [
            f'Dataset: {args.dataset}',
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
        ]
        pruning_param_name = 'num_minhash_funcs' if args.pruning_method == 'minhash_lsh' else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if args.pruning_method == 'minhash_lsh' else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name="GNN_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    for epoch in range(1, args.epochs + 1):
        loss, acc, f1 = train(epoch, dataset, train_loader, model, device, optimizer, tb_writer)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {f1:.4f}')

        train_acc, val_acc, test_acc = test(dataset, model, device)
        print(f'Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}, '
              f'Test ACC: {test_acc:.4f}')

        tb_writer.add_scalars('Accuracy',
                              {'train': train_acc,
                               'Validation': val_acc,
                               'Test': test_acc},
                              epoch)


if __name__ == '__main__':
    main()
