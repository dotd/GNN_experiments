import logging
import os
from time import time
from os import path
import argparse
from functools import partial
from typing import Dict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from src.utils.date_utils import get_time_str
from src.utils.graph_prune_utils import tg_dataset_prune
from src.utils.logging_utils import register_logger, log_args_description, get_clearml_logger
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash
from src.utils.proxy_utils import set_proxy
from tst.ogb.encoder_utils import augment_edge, decode_arr_to_seq, encode_y_to_arr, get_vocab_mapping
from tst.ogb.exp_utils import get_loss_function, evaluate, train
from tst.ogb.model_and_data_utils import add_zeros, create_model


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
                        help='dataset name (default: ogbg-molhiv)',
                        choices=['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2'])
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random',
                        choices=["minhash_lsh", "random"])
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--num_minhash_funcs', type=int, default=1)

    # dataset specific params:
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='Maximum sequence length to predict -- for ogbgb-code (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='The number of vocabulary used for sequence prediction (default: 5000)')

    # logging params:
    parser.add_argument('--exps_dir', type=str, help='Target directory to save logging files')
    parser.add_argument('--enable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="Enable logging to ClearML server")

    args = parser.parse_args()

    if args.enable_clearml_logger:
        clearml_logger = get_clearml_logger(project_name="GNN_pruning", task_name=f"pruning_method_{args.pruning_method}")

    tb_writer = None
    best_results_file = None
    log_file = None
    if args.exps_dir is not None:
        exps_dir = Path(args.exps_dir) / 'pyg_with_pruning' / args.dataset / args.pruning_method / str(args.random_pruning_prob)
        if args.pruning_method == 'minhash_lsh':
            exps_dir = exps_dir / str(args.num_minhash_funcs)

        exps_dir = exps_dir / get_time_str()
        best_results_file = exps_dir / 'best_results.txt'
        log_file = exps_dir / r'log.log'
        tensorboard_dir = exps_dir / 'tensorboard'
        if not tensorboard_dir.exists():
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        tb_writer.iteration = 0

    register_logger(log_file=log_file, stdout=True)
    log_args_description(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.proxy:
        set_proxy()

    # automatic data loading and splitting
    transform = add_zeros if args.dataset == 'ogbg-ppa' else None
    dataset = PygGraphPropPredDataset(name=args.dataset, transform=transform)
    cls_criterion = get_loss_function(dataset.name)
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
    logging.info(
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
        logging.info('using simple feature')
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
        logging.info(f'=====Epoch {epoch}')
        logging.info('Training...')
        train(model, device, train_loader, optimizer, cls_criterion=cls_criterion, tb_writer=tb_writer)

        logging.info('Evaluating...')
        train_perf = evaluate(model=model, device=device, loader=train_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset)
        valid_perf = evaluate(model=model, device=device, loader=valid_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset)
        test_perf = evaluate(model=model, device=device, loader=test_loader, evaluator=evaluator,
                             arr_to_seq=idx2word_mapper, dataset_name=args.dataset)

        logging.info({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if tb_writer is not None:
            tb_writer.add_scalars(dataset.eval_metric,
                                  {'Train': train_perf[dataset.eval_metric],
                                   'Validation': valid_perf[dataset.eval_metric],
                                   'Test': test_perf[dataset.eval_metric]},
                                  epoch)

    best_val_epoch = np.argmax(np.array(valid_curve)).item()
    best_train = max(train_curve)
    finish_time = time()
    logging.info(f'Finished training! Elapsed time: {(finish_time - start_time) / 60} minutes')
    logging.info('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    logging.info('Test score: {}'.format(test_curve[best_val_epoch]))

    if best_results_file is not None:
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, best_results_file)


if __name__ == "__main__":
    main()
