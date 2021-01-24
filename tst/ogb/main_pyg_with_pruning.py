from time import time

import argparse
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from src.utils.graph_prune_utils import tg_dataset_prune
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash
from src.utils.proxy_utils import set_proxy
from tst.ogb.encoder_utils import augment_edge, decode_arr_to_seq, encode_y_to_arr, get_vocab_mapping
from tst.ogb.model_and_data_utils import add_zeros, create_model


def train(model, device, loader, optimizer, cls_criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            # Treat single label and multi-label data differently.

            if hasattr(batch, 'y_arr'):
                loss = 0
                for i in range(len(pred)):
                    loss += cls_criterion(pred[i].to(torch.float32), batch.y_arr[:, i])
                loss = loss / len(pred)
            elif hasattr(batch, 'y'):
                # ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                raise AttributeError("Batch does not contain either a y-member or a y_arr-member")

            loss.backward()
            optimizer.step()


def evaluate(model, device, loader, evaluator, arr_to_seq, dataset_name: str):
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

            if dataset_name == 'ogbg-code':
                mat = []
                for i in range(len(pred)):
                    mat.append(torch.argmax(pred[i], dim=1).view(-1, 1))
                mat = torch.cat(mat, dim=1)
                seq_pred = [arr_to_seq(arr) for arr in mat]
                seq_ref = [batch.y[i] for i in range(len(batch.y))]
                y_true.extend(seq_ref)
                y_pred.extend(seq_pred)
            elif dataset_name == 'ogbg-molhiv':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                raise AttributeError("Batch does not contain either a y-member or a y_arr-member")

    if dataset_name == 'ogbg-code':
        input_dict = {"seq_ref": y_true, "seq_pred": y_pred}
    elif dataset_name == 'ogbg-molhiv':
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def get_prune_args(pruning_method: str, num_minhash_funcs: int, random_pruning_prob: float, node_dim: int) -> Dict:
    rnd = np.random.RandomState(0)
    if pruning_method == "minhash_lsh":
        minhash = MinHash(num_minhash_funcs, rnd, prime=2147483647)
        # MinHash parameters
        print(f"minhash:\n{minhash}")
        # LSH parameters
        lsh_num_funcs = 2
        sparsity = 3
        std_of_threshold = 1
        lsh = LSH(node_dim,
                  num_functions=lsh_num_funcs,
                  sparsity=sparsity,
                  std_of_threshold=std_of_threshold,
                  random=rnd)
        print(f"lsh:\n{lsh}")

        prune_args = {"minhash": minhash, "lsh": lsh}
    elif pruning_method == "random":
        prune_args = {"random": rnd, "p": random_pruning_prob}
    else:
        raise ValueError("Invalid pruning method")
    return prune_args


def main():
    start_time = time()
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

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random')
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--num_minhash_funcs', type=int, default=1)

    # dataset specific params:
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='Maximum sequence length to predict -- for ogbgb-code (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='The number of vocabulary used for sequence prediction (default: 5000)')

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.proxy:
        set_proxy()

    # automatic data loading and splitting
    transform = add_zeros if args.dataset == 'ogbg-ppa' else None
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=transform)
    cls_criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'ogbg-code' else torch.nn.BCEWithLogitsLoss()
    idx2word_mapper = None
    split_idx = dataset.get_idx_split()
    # The following is only used in the evaluation of the ogbg-code classifier.
    if args.dataset == 'ogbg-code':
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
        # specific transformations for the ogbg-code dataset
        dataset.transform = transforms.Compose(
            [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
        idx2word_mapper = partial(decode_arr_to_seq, idx2vocab=idx2vocab)

    # Get pruning arguments
    prune_args = get_prune_args(pruning_method=args.pruning_method, num_minhash_funcs=args.num_minhash_funcs,
                                random_pruning_prob=args.random_pruning_prob, node_dim=dataset[0].x.shape[1])

    train_data = list(dataset[split_idx["train"]])
    validation_data = list(dataset[split_idx["valid"]])
    test_data = list(dataset[split_idx["test"]])
    old_avg_edge_count = np.mean([g.edge_index.shape[1] for g in train_data])
    tg_dataset_prune(train_data, args.pruning_method, **prune_args)
    avg_edge_count = np.mean([g.edge_index.shape[1] for g in train_data])
    print(
        f"Old average number of edges: {old_avg_edge_count}. New one: {avg_edge_count}. Change: {(old_avg_edge_count - avg_edge_count) / old_avg_edge_count * 100}\%")
    tg_dataset_prune(validation_data, args.pruning_method, **prune_args)
    tg_dataset_prune(test_data, args.pruning_method, **prune_args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    evaluator = Evaluator(args.dataset)
    model = create_model(dataset=dataset, emb_dim=args.emb_dim,
                         dropout_ratio=args.drop_ratio, device=device, num_layers=args.num_layer,
                         max_seq_len=args.max_seq_len, num_vocab=args.num_vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print(f'=====Epoch {epoch}')
        print('Training...')
        train(model, device, train_loader, optimizer, cls_criterion=cls_criterion)

        print('Evaluating...')
        train_perf = evaluate(model=model, device=device, loader=train_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset)
        valid_perf = evaluate(model=model, device=device, loader=valid_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset)
        test_perf = evaluate(model=model, device=device, loader=test_loader, evaluator=evaluator,
                             arr_to_seq=idx2word_mapper, dataset_name=args.dataset)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    best_val_epoch = np.argmax(np.array(valid_curve)).item()
    best_train = max(train_curve)
    finish_time = time()
    print(f'Finished training! Elapsed time: {(finish_time - start_time) / 60} minutes')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
