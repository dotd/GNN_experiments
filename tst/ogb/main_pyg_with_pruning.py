import argparse
import json
import logging
from functools import partial
from pathlib import Path
from time import time
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels, ZINC, QM9
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from src.utils.date_utils import get_time_str
from src.utils.email_utils import GmailNotifier
from src.utils.evaluate import Evaluator
from src.utils.extract_target_transform import ExtractTargetTransform
from src.utils.graph_prune_utils import tg_dataset_prune
from src.utils.logging_utils import register_logger, log_args_description, get_clearml_logger, log_command
from src.utils.lsh_euclidean_tools import LSH
from src.utils.minhash_tools import MinHash, MinHashRep, MinHashRandomProj
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


def get_args():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=str, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gcn, or gcn-virtual (default: gcn)', )
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
                        choices=['github', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2', 'mnist', 'zinc',
                                 'reddit',
                                 'amazon_photo', 'amazon_comp', "Cora", "CiteSeer", "PubMed", 'QM9', 'ppi', 'flickr'])
    parser.add_argument('--target', type=int, default=0,
                        help='for datasets with multiple tasks, provide the target index')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random',
                        choices=["minhash_lsh_thresholding", "minhash_lsh_projection", "random"])
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--num_minhash_funcs', type=int, default=1)
    parser.add_argument('--sparsity', type=int, default=25)
    parser.add_argument("--complement", action='store_true', help="")
    parser.add_argument("--quantization_step", type=int, default=1, help="")


    # dataset specific params:
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='Maximum sequence length to predict -- for ogbgb-code (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='The number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--test', action="store_true", default=False,
                        help="Run in test mode", )
    parser.add_argument('--sample', type=float, default=1, help='The size of the sampled dataset')
    parser.add_argument('--line_graph', default=False, action='store_true',
                        help='Convert every graph G to L(G) as a line graph')

    # logging params:
    parser.add_argument('--csv_file', type=str, help='results.csv')
    parser.add_argument('--exps_dir', type=str, help='Target directory to save logging files')
    parser.add_argument('--enable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="Enable logging to ClearML server")
    parser.add_argument('--send_email', default=False, action='store_true', help='Send an email when finished')
    parser.add_argument('--email_user', default=r'eitan.kosman', help='Username for sending the email')
    parser.add_argument('--email_password', default='kqdopssgpcglbwaj', help='Password for sending the email')
    parser.add_argument('--email_to', default=r'eitan.kosman@gmail.com',
                        help='Email of the receiver of the results email')

    return parser.parse_args()


def register_logging_files(args):
    tb_writer = None
    best_results_file = None
    log_file = None
    if args.exps_dir is not None:
        exps_dir = Path(args.exps_dir) / 'pyg_with_pruning' / args.dataset / args.pruning_method
        if args.pruning_method == 'random':
            exps_dir = exps_dir / str(args.random_pruning_prob)
        elif args.pruning_method == 'minhash_lsh':
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
    log_command()
    log_args_description(args)

    clearml_task = None

    if args.enable_clearml_logger:
        tags = [
            f'Dataset: {args.dataset}',
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
        ]
        pruning_param_name = 'num_minhash_funcs' if 'minhash_lsh' in args.pruning_method else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if 'minhash_lsh' in args.pruning_method else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')

        if pruning_param_name == 'num_minhash_funcs':
            tags.append(f'Sparsity: {args.sparsity}')
            tags.append(f'Complement: {args.complement}')

        clearml_task = get_clearml_logger(f"GNN_{args.dataset}_{args.target}_{args.gnn}",
                                          task_name=get_time_str(),
                                          tags=tags)

    return tb_writer, best_results_file, log_file, clearml_task


def load_dataset(args):
    # automatic data loading and splitting
    transform = add_zeros if args.dataset == 'ogbg-ppa' else None
    cls_criterion = get_loss_function(args.dataset)
    idx2word_mapper = None

    if args.dataset == 'mnist':
        train_data = MNISTSuperpixels(root='dataset', train=True, transform=T.Polar())
        dataset = train_data
        dataset.name = 'mnist'
        dataset.eval_metric = 'acc'
        validation_data = []
        test_data = MNISTSuperpixels(root='dataset', train=False, transform=T.Polar())

        train_data = list(train_data)
        test_data = list(test_data)

    elif args.dataset == 'QM9':
        # Contains 19 targets. Use only the first 12 (0-11)
        QM9_VALIDATION_START = 110000
        QM9_VALIDATION_END = 120000
        dataset = QM9(root='dataset', transform=ExtractTargetTransform(args.target)).shuffle()
        dataset.name = 'QM9'
        dataset.eval_metric = 'mae'

        train_data = dataset[:QM9_VALIDATION_START]
        validation_data = dataset[QM9_VALIDATION_START:QM9_VALIDATION_END]
        test_data = dataset[QM9_VALIDATION_END:]

        train_data = list(train_data)
        validation_data = list(validation_data)
        test_data = list(test_data)

    elif args.dataset == 'zinc':
        train_data = ZINC(root='dataset', subset=True, split='train')

        dataset = train_data
        dataset.name = 'zinc'
        validation_data = ZINC(root='dataset', subset=True, split='val')
        test_data = ZINC(root='dataset', subset=True, split='test')
        dataset.eval_metric = 'mae'

        train_data = list(train_data)
        validation_data = list(validation_data)
        test_data = list(test_data)

    elif args.dataset in ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name=args.dataset, transform=transform)

        if args.dataset == 'obgb-code2':
            seq_len_list = np.array([len(seq) for seq in dataset.data.y])
            max_seq_len = args.max_seq_len
            num_less_or_equal_to_max = np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)
            print(f'Target sequence less or equal to {max_seq_len} is {num_less_or_equal_to_max}%.')

        split_idx = dataset.get_idx_split()
        # The following is only used in the evaluation of the ogbg-code classifier.
        if args.dataset == 'ogbg-code2':
            vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
            # specific transformations for the ogbg-code dataset
            dataset.transform = transforms.Compose(
                [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
            idx2word_mapper = partial(decode_arr_to_seq, idx2vocab=idx2vocab)

        train_data = list(dataset[split_idx["train"]])
        validation_data = list(dataset[split_idx["valid"]])
        test_data = list(dataset[split_idx["test"]])

    return dataset, train_data, validation_data, test_data, cls_criterion, idx2word_mapper


def prune_datasets(train_data, validation_data, test_data, args):
    logging.info("Pruning datasets...")

    logging.info("Pruning training data...")
    pruning_params = prune_dataset(train_data, args)

    logging.info("Pruning validation data...")
    prune_dataset(validation_data, args, pruning_params=pruning_params)

    logging.info("Pruning test data...")
    prune_dataset(test_data, args, pruning_params=pruning_params)

    return train_data, validation_data, test_data


def prune_dataset(original_dataset, args, random=np.random.RandomState(0), pruning_params=None):
    if original_dataset is None or len(original_dataset) == 0:
        return None
    if args.pruning_method == 'minhash_lsh_projection':
        dim_nodes = original_dataset[0].x.shape[1] if len(original_dataset[0].x.shape) == 2 else 0
        std_of_threshold = 1
        mean_of_threshold = 1
        dim_edges = 0
        if original_dataset[0].edge_attr is not None:
            if len(original_dataset[0].edge_attr.shape) == 1:
                dim_edges = 1
            else:
                dim_edges = original_dataset[0].edge_attr.shape[1]

        din = 0
        if dim_nodes != 0:
            din += dim_nodes * 2
        if dim_edges != 0:
            din += dim_edges

        minhash_lsh = MinHashRandomProj(N=args.num_minhash_funcs,
                                        random=random,
                                        sparsity=args.sparsity,
                                        din=din,
                                        quantization_step=args.quantization_step)

        prunning_ratio = tg_dataset_prune(tg_dataset=original_dataset,
                                          method="minhash_lsh_projection",
                                          minhash=minhash_lsh, )
        print(f"prunning_ratio = {prunning_ratio}")

    elif args.pruning_method == 'minhash_lsh_thresholding':
        if pruning_params is None:
            dim_nodes = original_dataset[0].x.shape[1] if len(original_dataset[0].x.shape) == 2 else 0
            lsh_num_funcs = args.num_minhash_funcs
            sparsity = args.sparsity
            std_of_threshold = 1
            mean_of_threshold = 1
            dim_edges = 0
            if original_dataset[0].edge_attr is not None:
                if len(original_dataset[0].edge_attr.shape) == 1:
                    dim_edges = 1
                else:
                    dim_edges = original_dataset[0].edge_attr.shape[1]

            minhash = MinHashRep(lsh_num_funcs, random, prime=2147483647)
            pruning_params = {
                "minhash": minhash,
                'nodes': {'din': dim_nodes,
                          'num_functions': lsh_num_funcs,
                          'sparsity': sparsity,
                          'std_of_threshold': std_of_threshold,
                          'mean_of_threshold': mean_of_threshold,
                          'random': random, },
                'edges': {'din': dim_edges,
                          'num_functions': lsh_num_funcs,
                          'sparsity': sparsity,
                          'std_of_threshold': std_of_threshold,
                          'mean_of_threshold': mean_of_threshold,
                          'random': random, },
            }
            lsh_nodes = LSH(**pruning_params['nodes']) if dim_nodes != 0 else None
            lsh_edges = LSH(**pruning_params['edges']) if dim_edges != 0 else None

            pruning_params['nodes']['lsh'] = lsh_nodes
            pruning_params['edges']['lsh'] = lsh_edges

        print(f"lsh_nodes:\n{pruning_params['nodes']['lsh']}")
        print(f"lsh_edges:\n{pruning_params['edges']['lsh']}")

        prunning_ratio = tg_dataset_prune(tg_dataset=original_dataset,
                                          method="minhash_lsh_thresholding",
                                          minhash=pruning_params['minhash'],
                                          lsh_nodes=pruning_params['nodes']['lsh'],
                                          lsh_edges=pruning_params['edges']['lsh'],
                                          complement=args.complement)
        print(f"prunning_ratio = {prunning_ratio}")

    elif args.pruning_method == 'random':
        prunning_ratio = args.random_pruning_prob
        tg_dataset_prune(tg_dataset=original_dataset,
                         method="random",
                         p=args.random_pruning_prob,
                         random=random,
                         complement=args.complement)

    else:
        raise NotImplementedError(f"Pruning method {args.pruning_method} not implemented")

    return pruning_params, prunning_ratio


def get_optimizer_and_scheduler(args, model):
    """
    Returns an pytorch optimizer and scheduler for each specific dataset
    Args:
        args: arguments of the program
        model: the model we use for training and testing

    Returns: optimizer, scheduler

    """

    if args.dataset == 'QM9':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        scheduler_ = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    return optimizer, scheduler


def main():
    start_time = time()
    # Training settings
    args = get_args()

    tb_writer, best_results_file, log_file, clearml_task = register_logging_files(args)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = torch.device(
            "cuda:" + str(args.device)) if torch.cuda.is_available() and args.device != 'cpu' else torch.device("cpu")

    if args.proxy:
        set_proxy()

    dataset, train_data, validation_data, test_data, cls_criterion, idx2word_mapper = load_dataset(args)

    if args.test:
        train_data = train_data[:100]
        validation_data = validation_data[:100]
        test_data = test_data[:100]
    elif args.sample != 1:
        logging.info(f"Sampling {args.sample * 100}% of the dataset")
        train_idx = list(np.random.choice(len(train_data), int(len(train_data) * args.sample), replace=False))
        val_idx = list(np.random.choice(len(validation_data), int(len(validation_data) * args.sample), replace=False))
        test_idx = list(np.random.choice(len(test_data), int(len(test_data) * args.sample), replace=False))

        train_data = [train_data[i] for i in train_idx]
        validation_data = [validation_data[i] for i in val_idx]
        test_data = [test_data[i] for i in test_idx]

    old_avg_edge_count = np.mean([g.edge_index.shape[1] for g in train_data])

    # Prune the train data and cache the parameters for further usage
    train_data, validation_data, test_data = prune_datasets(train_data, validation_data, test_data, args)

    edges_per_sample = [idx.shape[1] for idx in (sample.edge_index for sample in train_data if sample.num_edges > 0)]
    avg_edge_count = np.mean(edges_per_sample) if len(edges_per_sample) != 0 else 0
    logging.info(
        f"Old average number of edges: {old_avg_edge_count}. New one: {avg_edge_count}. Change: {(old_avg_edge_count - avg_edge_count) / old_avg_edge_count * 100}\%")

    prunning_ratio = avg_edge_count / old_avg_edge_count

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
                         max_seq_len=args.max_seq_len, num_vocab=args.num_vocab, model_type=args.gnn)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    valid_curve = []
    test_curve = []
    train_curve = []

    train_times = []
    val_times = []

    for epoch in range(1, args.epochs + 1):
        logging.info(f'=====Epoch {epoch}')
        logging.info('Training...')
        seconds_per_iter = train(model, dataset, device, train_loader, optimizer, cls_criterion=cls_criterion,
                                 tb_writer=tb_writer)

        if scheduler is not None:
            scheduler.step(epoch)

        logging.info('Evaluating...')
        train_perf = evaluate(model=model, device=device, loader=train_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset, return_avg_time=False)
        valid_perf = evaluate(model=model, device=device, loader=valid_loader, evaluator=evaluator,
                              arr_to_seq=idx2word_mapper, dataset_name=args.dataset, return_avg_time=False)
        test_perf, test_seconds_per_iter = evaluate(model=model, device=device, loader=test_loader,
                                                    evaluator=evaluator,
                                                    arr_to_seq=idx2word_mapper, dataset_name=args.dataset,
                                                    return_avg_time=True)

        train_times.append(seconds_per_iter)
        val_times.append(test_seconds_per_iter)

        logging.info({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])

        if valid_perf is not None:
            valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if tb_writer is not None:
            if valid_perf is not None:
                tb_writer.add_scalars(dataset.eval_metric,
                                      {'Train': train_perf[dataset.eval_metric],
                                       'Validation': valid_perf[dataset.eval_metric],
                                       'Test': test_perf[dataset.eval_metric]},
                                      epoch)
            else:
                tb_writer.add_scalars(dataset.eval_metric,
                                      {'Train': train_perf[dataset.eval_metric],
                                       'Test': test_perf[dataset.eval_metric]},
                                      epoch)

    if dataset.eval_metric in ['rmse', 'mae']:
        best_val_epoch = np.argmin(np.array(valid_curve)).item() if valid_perf is not None else np.argmax(
            np.array(test_curve)).item()
        best_train = min(train_curve)
    else:
        best_val_epoch = np.argmax(np.array(valid_curve)).item() if valid_perf is not None else np.argmax(
            np.array(test_curve)).item()
        best_train = max(train_curve)
    finish_time = time()
    logging.info(f'Finished training! Elapsed time: {(finish_time - start_time) / 60} minutes')
    if valid_perf is not None:
        logging.info('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    logging.info('Test score: {}'.format(test_curve[best_val_epoch]))

    best_results = {'Train': train_curve[best_val_epoch],
                    'Test': test_curve[best_val_epoch],
                    'BestTrain': best_train,
                    r'training sec/iter': seconds_per_iter,
                    r'test sec/iter': test_seconds_per_iter}
    if valid_perf is not None:
        best_results['Val'] = valid_curve[best_val_epoch]

    if best_results_file is not None:
        with open(best_results_file, 'w') as fp:
            json.dump(best_results, fp)

    experiment_logs = dict()
    experiment_logs = clearml_task.connect(experiment_logs)
    experiment_logs['time/train'] = np.mean(train_times)
    experiment_logs['time/val'] = np.mean(val_times)
    experiment_logs['keep edges'] = prunning_ratio
    experiment_logs[f'best train {dataset.eval_metric}'] = train_curve[best_val_epoch]
    if valid_perf is not None:
        experiment_logs[f'best val {dataset.eval_metric}'] = valid_curve[best_val_epoch]
    experiment_logs[f'best test {dataset.eval_metric}'] = test_curve[best_val_epoch]

    if args.send_email:
        with GmailNotifier(username=args.email_user, password=args.email_password, to=args.email_to) as noti:
            noti.send_results('exps_pyg_with_pruning', args, best_results)


if __name__ == "__main__":
    main()
