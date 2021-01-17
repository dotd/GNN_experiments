import argparse

import numpy as np
import torch
import torch.optim as optim
# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src.utils.graph_prune_utils import tg_dataset_prune
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash
from src.utils.proxy_utils import set_proxy
from tst.ogb.gcn import GCN

cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gcn, or gcn-virtual (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )
    parser.add_argument('--pruning_method', type=str, default='random')
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.proxy:
        set_proxy()

    rnd = np.random.RandomState(0)

    if args.pruning_method == "minhash_lsh":
        num_minhash_funcs = 1
        minhash = MinHash(num_minhash_funcs, rnd, prime=2147483647)
        # MinHash parameters
        print(f"minhash:\n{minhash}")

        # LSH parameters
        lsh_num_funcs = 2
        sparsity = 3
        std_of_threshold = 1
        dim_nodes = 9
        lsh = LSH(dim_nodes,
                  num_functions=lsh_num_funcs,
                  sparsity=sparsity,
                  std_of_threshold=std_of_threshold,
                  random=rnd)
        print(f"lsh:\n{lsh}")

        prune_args = {"minhash": minhash, "lsh": lsh}
    elif args.pruning_method == "random":
        prune_args = {"random": rnd, "p": args.random_pruning_prob}
    else:
        raise ValueError("Invalid pruning method")

    # automatic data loading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    train_data = list(dataset[split_idx["train"]])
    validation_data = list(dataset[split_idx["valid"]])
    test_data = list(dataset[split_idx["test"]])

    # orig_train_data_num_edges = train_data.data.edge_index.shape[1]
    old_avg_edge_count = np.mean([g.edge_index.shape[1] for g in train_data])
    tg_dataset_prune(train_data, args.pruning_method, **prune_args)
    avg_edge_count = np.mean([g.edge_index.shape[1] for g in train_data])
    print(
        f"Old average number of edges: {old_avg_edge_count}. New one: {avg_edge_count}. Change: {(old_avg_edge_count - avg_edge_count) / old_avg_edge_count * 100}\%")
    tg_dataset_prune(validation_data, args.pruning_method, **prune_args)
    tg_dataset_prune(test_data, args.pruning_method, **prune_args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = GCN(num_tasks=dataset.num_tasks, num_layer=args.num_layer,
                emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print(f"=====Epoch {epoch}")
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = evaluate(model, device, train_loader, evaluator)
        valid_perf = evaluate(model, device, valid_loader, evaluator)
        test_perf = evaluate(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print(f'Best validation score: {valid_curve[best_val_epoch]}')
    print(f'Test score in max-validation score epoch: {test_curve[best_val_epoch]}')

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
