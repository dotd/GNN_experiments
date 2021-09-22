import argparse

import numpy as np
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv, GCNConv

from src.utils.csv_utils import prepare_csv
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.ogb.main_pyg_with_pruning import prune_datasets, prune_dataset

start_time = time.time()

import src.synthetic.random_graph_dataset as rgd
from tst.torch_geometric.tst_torch_geometric1 import GCN, GAT

import src.synthetic.synthetic_utils as su
from tst.torch_geometric.tst_torch_geometric1 import train, func_test
from torch_geometric.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=str, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gcn, or gcn-virtual (default: gcn)',
                        choices=['gcn', 'gat', 'monet', 'pna', 'sage', 'mlp', 'mxmnet'])
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="synthetic",
                        help='dataset name (default: ogbg-molhiv)',
                        choices=['synthetic'])
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random', )
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--num_minhash_funcs', type=int, default=1)
    parser.add_argument('--sparsity', type=int, default=8)
    parser.add_argument("--complement", action='store_true', help="")
    parser.add_argument("--quantization_step", type=int, default=1, help="")

    # logging params:
    parser.add_argument('--exps_dir', type=str, help='Target directory to save logging files')
    parser.add_argument('--csv_file', type=str, default='synthetic_results_tmp.csv')

    parser.add_argument('--enable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="Enable logging to ClearML server")
    parser.add_argument('--send_email', default=False, action='store_true', help='Send an email when finished')
    parser.add_argument('--email_user', default=r'eitan.kosman', help='Username for sending the email')
    parser.add_argument('--email_password', default='kqdopssgpcglbwaj', help='Password for sending the email')
    parser.add_argument('--email_to', default=r'eitan.kosman@gmail.com',
                        help='Email of the receiver of the results email')

    # dataset specific params:
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--generate_only', default=False, action='store_true',
                        help='whether to only generate the dataset and exit')

    parser.add_argument('--num_samples', type=int, default=20000, help='')
    parser.add_argument('--num_classes', type=int, default=100, help='')
    parser.add_argument('--min_nodes', type=int, default=40, help='')
    parser.add_argument('--max_nodes', type=int, default=60, help='')
    parser.add_argument('--dim_nodes', type=int, default=10, help='')
    parser.add_argument('--dim_edges', type=int, default=40, help='')
    parser.add_argument('--connectivity_rate', type=float, default=0.2,

                        help='how many edges are connected to each node, normalized')
    parser.add_argument('--centers_nodes_std', type=float, default=0.2, help='the std of the nodes representation')
    parser.add_argument('--centers_edges_std', type=float, default=0.2, help='the std of the edges representation')

    parser.add_argument('--node_additive_noise_std', type=float, default=0.25,

                        help='the std of the nodes noise, per sample')
    parser.add_argument('--edge_additive_noise_std', type=float, default=0.1,
                        help='the std of the edges noise, per sample')

    parser.add_argument('--connectivity_rate_noise', type=float, default=0.1,
                        help='the edges rate for the parallel graph')
    parser.add_argument('--symmetric_flag', default=False, action='store_true',
                        help='whether the graph is bidirectional')
    parser.add_argument('--nodes_order_scramble_flag', default=False, action='store_true',
                        help='whether to rename nodes')
    parser.add_argument('--noise_remove_node', type=float, default=0.1,
                        help='the rate of removing nodes from the generated graph')
    parser.add_argument('--noise_add_node', type=float, default=0.1,
                        help='the rate of adding noisy nodes to the generated graph, each new node will be connected'
                             'with the original connectivity rate')  # TODO: 1) how to generate edge and node features

    return parser.parse_args()


def get_model(arch, dim_nodes, num_classes, num_hidden=40):
    if arch == 'gcn':
        model = GCN(hidden_channels=num_hidden, in_size=dim_nodes, out_size=num_classes, conv_ctr=GCNConv)
    elif arch == 'gat':
        model = GAT(dim_nodes, num_classes, heads=16, num_hidden=num_hidden)

    return model

  
def tst_classify_networkx_synthetic_tg(
        args,
        num_samples=1000,
        num_classes=2,
        min_nodes=10,
        max_nodes=10,
        dim_nodes=4,
        dim_edges=4,
        centers_nodes_std=0.1,
        centers_edges_std=0.1,
        connectivity_rate=0.2,
        connectivity_rate_noise=0.05,
        symmetric_flag=True,
        nodes_order_scramble_flag=True,
        node_additive_noise_std=0.1,
        edge_additive_noise_std=0.1,
        random=np.random.RandomState(0),
        noise_remove_node=0.1,
        noise_add_node=0.1,
        tb_writer=None,
        graph_dataset=None,
        **kwargs,
):
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")

    if graph_dataset is None:
        graph_dataset = rgd.generate_graphs_dataset(num_samples=num_samples,
                                                    num_classes=num_classes,
                                                    min_nodes=min_nodes,
                                                    max_nodes=max_nodes,
                                                    dim_nodes=dim_nodes,
                                                    dim_edges=dim_edges,
                                                    centers_nodes_std=centers_nodes_std,
                                                    centers_edges_std=centers_edges_std,
                                                    connectivity_rate=connectivity_rate,
                                                    connectivity_rate_noise=connectivity_rate_noise,
                                                    noise_remove_node=noise_remove_node,
                                                    node_additive_noise_std=node_additive_noise_std,
                                                    edge_additive_noise_std=edge_additive_noise_std,
                                                    noise_add_node=noise_add_node,
                                                    nodes_order_scramble_flag=nodes_order_scramble_flag,
                                                    symmetric_flag=symmetric_flag,
                                                    random=random)

    tg_dataset = su.transform_dataset_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels)
    print(f"{time.time() - start_time:.4f} Finished generating dataset")

    pruning_params, prunning_ratio = prune_dataset(tg_dataset, args)

    tg_dataset_train, tg_dataset_test = train_test_split(tg_dataset, test_size=0.25)

    train_loader = DataLoader(tg_dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(tg_dataset_test, batch_size=args.batch_size, shuffle=False)

    model = get_model(arch=args.gnn, dim_nodes=dim_nodes, num_classes=num_classes).to(args.device)
    test_acc, _ = func_test(args, model, test_loader)

    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    best_train = 0
    best_test = 0
    train_times = []
    test_times = []
    for epoch in range(args.epochs):
        avg_time_train = train(args, model, train_loader)
        train_times.append(avg_time_train)
        train_acc, _ = func_test(args, model, train_loader)
        test_acc, avg_time_test = func_test(args, model, test_loader)

        test_times.append(avg_time_test)
        best_train = max(best_train, train_acc)
        best_test = max(best_test, test_acc)

        if tb_writer is not None:
            tb_writer.add_scalars('Accuracy',
                                  {'Train': train_acc,
                                   'Test': test_acc, },
                                  epoch)

        print(
            f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return graph_dataset, prunning_ratio, best_train, best_test, np.mean(train_times), np.mean(test_times)



def main(args):
    vals = dict()
    csv_file = args.csv_file
    """
    Pruning with LSH
    """
    args.pruning_method = 'minhash_lsh_projection'

    tb_writer = None
    if args.enable_clearml_logger:
        tb_writer = SummaryWriter(log_dir=None)
        tags = [
            f'Pruning method: {args.pruning_method}',
            f'Architecture: {args.gnn}',
            f'dim_edges:{args.dim_edges}',
        ]
        pruning_param_name = 'num_minhash_funcs' if 'minhash_lsh' in args.pruning_method else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if 'minhash_lsh' in args.pruning_method else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name="GNN_synthetic_pruning_dimensionality",

                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    graph_dataset, prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = \
        tst_classify_networkx_synthetic_tg(**vars(args), tb_writer=tb_writer, args=args, graph_dataset=None)
    print(f"{time.time() - start_time:.4f} end time")

    vals['keep edges'] = prunning_ratio
    vals['minhash train'] = best_train
    vals['minhash test'] = best_test

    vals['minhash time train'] = avg_time_train
    vals['minhash time test'] = avg_time_test

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
        pruning_param_name = 'num_minhash_funcs' if 'minhash_lsh' in args.pruning_method else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if 'minhash_lsh' in args.pruning_method else args.random_pruning_prob

        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name="GNN_synthetic_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    graph_dataset, prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = \
        tst_classify_networkx_synthetic_tg(**vars(args),
                                           tb_writer=tb_writer,
                                           args=args,
                                           graph_dataset=graph_dataset)
    print(f"{time.time() - start_time:.4f} end time")
    vals['random train'] = best_train
    vals['random test'] = best_test
    vals['random time train'] = avg_time_train
    vals['random time test'] = avg_time_test
    vals['architecture'] = args.gnn

    df = pd.read_csv(csv_file)
    df = df.append(vals, ignore_index=True)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    main(get_args())
