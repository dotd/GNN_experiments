import argparse
import enum
import time
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.archs.ppi_gat.ppi_gat import GAT, LayerType
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.utils.ppi_data_loading import load_graph_data


# Implementation from https://github.com/gordicaleksa/pytorch-GAT


class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


def save_best_stats(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_main_loop(args, gat, sigmoid_cross_entropy_loss, optimizer, patience_period, time_start, tb_writer):
    device = next(gat.parameters()).device  # fetch the device info from the model instead of passing it as a param

    @save_best_stats(best_train_perf=0, best_val_perf=0, best_val_loss=0, patience_cnt=0, best_test_perf=0)
    def main_loop(phase, data_loader, epoch=0):

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
        # We merge them into a single graph with 2 connected components, that's the main idea. After that
        # the implementation #3 is agnostic to the fact that those are multiple and not a single graph!
        current_start_time = time.time()
        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
            edge_index = edge_index.to(device)
            node_features = node_features.to(device)
            gt_node_labels = gt_node_labels.to(device)

            # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
            # shape = (N, C) where N is the number of nodes in the batch and C is the number of classes (121 for PPI)
            # nodes_unnormalized_scores = gat(node_features, edge_index)
            nodes_unnormalized_scores = gat((node_features, edge_index))[0]

            loss = sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

            if phase == LoopPhase.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate the main metric - micro F1
            pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            # Logging
            global_step = len(data_loader) * epoch + batch_idx
            if phase == LoopPhase.TRAIN:
                main_loop.best_train_perf = max(micro_f1, main_loop.best_train_perf)
                # Log metrics
                if args.enable_tensorboard:
                    tb_writer.add_scalar('training_loss', loss.item(), global_step)
                    tb_writer.add_scalar('training_micro_f1', micro_f1, global_step)

                # Log to console
                if args.console_log_freq is not None and batch_idx % args.console_log_freq == 0:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | train micro-F1={micro_f1} | {(time.time() - current_start_time) / (batch_idx + 1)} seconds/iteration')

            elif phase == LoopPhase.VAL:
                # Log metrics
                if args.enable_tensorboard:
                    tb_writer.add_scalar('val_loss', loss.item(), global_step)
                    tb_writer.add_scalar('val_micro_f1', micro_f1, global_step)

                # Log to console
                if args.console_log_freq is not None and batch_idx % args.console_log_freq == 0:
                    print(f'GAT validation: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | val micro-F1={micro_f1} | {(time.time() - current_start_time) / (batch_idx + 1)} seconds/iteration')

                # The "patience" logic - should we break out from the training loop? If either validation micro-F1
                # keeps going up or the val loss keeps going down we won't stop
                if micro_f1 > main_loop.best_val_perf or loss.item() < main_loop.best_val_loss:
                    main_loop.best_val_perf = max(micro_f1, main_loop.best_val_perf)  # keep track of the best validation micro_f1 so far
                    main_loop.best_val_loss = min(loss.item(), main_loop.best_val_loss)  # and the minimal loss
                    main_loop.patience_cnt = 0  # reset the counter every time we encounter new best micro_f1
                else:
                    main_loop.patience_cnt += 1  # otherwise keep counting

                if main_loop.patience_cnt >= patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

            else:
                main_loop.best_test_perf = max(main_loop.best_test_perf, micro_f1)
                return micro_f1  # in the case of test phase we just report back the test micro_f1

        return (time.time() - current_start_time) / len(data_loader)

    return main_loop  # return the decorated function


pruning_ratio_path = 'last_pruning_ratio.txt'


def train_gat_ppi(args, tb_writer, clearml_task):
    """
    Very similar to Cora's training script. The main differences are:
    1. Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    2. Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    3. Model architecture and hyperparams are a bit different (as reported in the GAT paper)
    """
    # if pathlib.Path(pruning_ratio_path).exists():
    #     with open(pruning_ratio_path, 'r') as fp:
    #         args.random_pruning_prob = float(fp.read())

    # Checking whether you have a strong GPU. Since PPI training requires almost 8 GBs of VRAM
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    device = torch.device(args.device if torch.cuda.is_available() and not args.force_cpu else "cpu")

    # Step 1: prepare the data loaders
    data_loader_train, data_loader_val, data_loader_test, prune_ratio = load_graph_data(args, device)

    with open(pruning_ratio_path, 'w') as fp:
        fp.write(str(prune_ratio))

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=args.num_of_layers,
        num_heads_per_layer=args.num_heads_per_layer,
        num_features_per_layer=args.num_features_per_layer,
        add_skip_connection=args.add_skip_connection,
        bias=args.bias,
        dropout=args.dropout,
        layer_type=args.layer_type,
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    # gat = NodeGCN(num_features=args.num_features_per_layer[0],
    #               num_classes=args.num_features_per_layer[-1],
    #               num_hidden=args.num_features_per_layer[1],
    #               num_layers=3,
    #               apply_log_softmax=False).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        args,
        gat,
        loss_fn,
        optimizer,
        args.patience_period,
        time.time(),
        tb_writer, )

    # Step 4: Start the training procedure
    train_times = []
    val_times = []
    max_train_acc = 0
    max_val_acc = 0
    max_test_acc = 0
    for epoch in range(args.num_of_epochs):
        # Training loop
        train_time = main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)
        train_times.append(train_time)
        # Validation loop
        with torch.no_grad():
            try:
                val_time = main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
                val_times.append(val_time)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

        # Step 5: Test the model
        # if args.should_test:
        micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)

        print('*' * 50)
        print(f'Test micro-F1 = {main_loop.best_test_perf}')

    print(f"Pruning ratio: {prune_ratio}")
    tb_writer.add_scalar('time/train', np.mean(train_times), 0)
    tb_writer.add_scalar('time/val', np.mean(val_times), 0)
    tb_writer.add_scalar('test', micro_f1, 0)

    experiment_logs = dict()
    experiment_logs = clearml_task.connect(experiment_logs)
    experiment_logs['time/train'] = np.mean(train_times)
    experiment_logs['time/val'] = np.mean(val_times)
    experiment_logs['keep edges'] = prune_ratio
    experiment_logs['max train accuracy'] = main_loop.best_train_perf
    experiment_logs['max val accuracy'] = main_loop.best_val_perf
    experiment_logs['test accuracy'] = main_loop.best_test_perf


def get_training_args():
    ppi_num_input_features = 50
    ppi_num_classes = 121
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=100)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--force_cpu", action='store_true', help='use CPU if your GPU is too small (no by default)')
    parser.add_argument("--device", type=str, help='')

    # GAT configs
    # parser.add_argument("--num_of_layers", type=int, help='', default=3)
    parser.add_argument('--gnn', type=str, default='gat',
                        help='GNN gcn, or gcn-virtual (default: gcn)',)
    parser.add_argument("--num_of_layers", type=list, help='', default=[4, 4, 6])
    parser.add_argument("--num_features_per_layer", type=list, help='', default=[ppi_num_input_features, 256, 256, ppi_num_classes])
    parser.add_argument("--add_skip_connection", type=bool, help='', default=True)
    parser.add_argument("--bias", type=bool, help='', default=True)
    parser.add_argument("--dropout", type=float, help='', default=0.0)
    parser.add_argument("--layer_type", help='', default=LayerType.IMP3)

    # Dataset related (note: we need the dataset name for metadata and related stuff, and not for picking the dataset)
    parser.add_argument("--dataset_name", choices=['PPI'], help='dataset to use for training', default='PPI')
    parser.add_argument("--batch_size", type=int, help='number of graphs in a batch', default=2)
    parser.add_argument("--ppi_load_test_only", type=bool, default=False, help='')

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random',
                        choices=["minhash_lsh_thresholding", "minhash_lsh_projection", "random"])
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--num_minhash_funcs', type=int, default=1)
    parser.add_argument('--sparsity', type=int, default=25)
    parser.add_argument("--complement", action='store_true', help="")
    parser.add_argument("--quantization_step", type=int, default=1, help="")

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq (None for no logging)", default=1)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=5)
    parser.add_argument('--enable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="Enable logging to ClearML server")
    args = parser.parse_args()

    # I'm leaving the hyperparam values as reported in the paper, but I experimented a bit and the comments suggest
    # how you can make GAT achieve an even higher micro-F1 or make it smaller
    gat_config = {
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [ppi_num_input_features, 256, 256, ppi_num_classes],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
        "layer_type": LayerType.IMP3  # the only implementation that supports the inductive setting
    }

    for k, v in gat_config.items():
        setattr(args, k, v)

    # Wrapping training configuration into a dictionary
    # training_config = dict()
    # for arg in vars(args):
    #     training_config[arg] = getattr(args, arg)
    tb_writer = None
    clearml_logger = None
    if args.enable_clearml_logger:
        args.enable_tensorboard = True
        tb_writer = SummaryWriter()
        tags = [
            f'Dataset: {args.dataset_name}',
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
        ]
        pruning_param_name = 'num_minhash_funcs' if 'minhash_lsh' in args.pruning_method else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if 'minhash_lsh' in args.pruning_method else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')

        if pruning_param_name == 'num_minhash_funcs':
            tags.append(f'Sparsity: {args.sparsity}')
            tags.append(f'Complement: {args.complement}')

        clearml_logger = get_clearml_logger(project_name=f"GNN_PPI_{args.gnn}",
                                            task_name=get_time_str(),
                                            tags=tags)

    return args, tb_writer, clearml_logger


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    args, tb_writer, clearml_logger = get_training_args()
    train_gat_ppi(args, tb_writer, clearml_logger)
