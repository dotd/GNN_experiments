import argparse

import numpy as np
import time

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.ogb.main_pyg_with_pruning import prune_datasets, prune_dataset

start_time = time.time()

import src.synthetic.random_graph_dataset as rgd
from tst.torch_geometric.tst_torch_geometric1 import GCN
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)',
                        choices=['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2', 'mnist', 'zinc', 'reddit',
                                 'amazon_comp', "Cora", "CiteSeer", "PubMed", 'QM9'])
    parser.add_argument('--target', type=int, default=0,
                        help='for datasets with multiple tasks, provide the target index')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )

    # Pruning specific params:
    parser.add_argument('--pruning_method', type=str, default='random',
                        choices=["minhash_lsh", "random"])
    parser.add_argument('--random_pruning_prob', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--num_minhash_funcs', type=int, default=1)
    parser.add_argument('--sparsity', type=int, default=25)
    parser.add_argument("--complement", action='store_true', help="")

    # logging params:
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

    # dataset specific params:
    parser.add_argument('--num_samples', type=int, default=25000, help='')
    parser.add_argument('--num_classes', type=int, default=50, help='')
    parser.add_argument('--min_nodes', type=int, default=50, help='')
    parser.add_argument('--max_nodes', type=int, default=60, help='')
    parser.add_argument('--dim_nodes', type=int, default=10, help='')
    parser.add_argument('--dim_edges', type=int, default=10, help='')
    parser.add_argument('--connectivity_rate', type=float, default=0.2,
                        help='how many edges are connected to each node, normalized')
    parser.add_argument('--centers_nodes_std', type=float, default=0.2, help='the std of the nodes representation')
    parser.add_argument('--centers_edges_std', type=float, default=0.2, help='the std of the edges representation')

    parser.add_argument('--node_additive_noise_std', type=float, default=0.1,
                        help='the std of the nodes noise, per sample')
    parser.add_argument('--edge_additive_noise_std', type=float, default=0.1,
                        help='the std of the edges noise, per sample')

    parser.add_argument('--connectivity_rate_noise', type=float, default=0.05,
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
        **kwargs,
):
    print(f"{time.time() - start_time:.4f} tst_classify_synthetic")

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
    # rgd.graph_sample_dataset_to_networkx(graph_dataset)
    tg_dataset = su.transform_dataset_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels)
    print(f"{time.time() - start_time:.4f} Finished generating dataset")

    # print("")
    # print(graph_dataset)
    # tg_dataset = su.transform_networkx_to_torch_geometric_dataset(graph_dataset.samples, graph_dataset.labels)

    pruning_params = prune_dataset(tg_dataset, args)

    tg_dataset_train, tg_dataset_test = train_test_split(tg_dataset, test_size=0.25)

    train_loader = DataLoader(tg_dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(tg_dataset_test, batch_size=args.batch_size, shuffle=False)

    model = GCN(hidden_channels=60, in_size=dim_nodes, out_size=num_classes)
    test_acc, _ = func_test(model, test_loader)
    print(f'{time.time() - start_time:.4f} Test Acc: {test_acc:.4f}')

    for epoch in range(args.epochs):
        train(model, train_loader)
        train_acc, _ = func_test(model, train_loader)
        test_acc, _ = func_test(model, test_loader)

        if tb_writer is not None:
            tb_writer.add_scalars('Accuracy',
                                  {'Train': train_acc,
                                   'Test': test_acc, },
                                  epoch)

        print(
            f'{time.time() - start_time:.4f} Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def main(args):
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
        clearml_logger = get_clearml_logger(project_name="GNN_synthetic_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    tst_classify_networkx_synthetic_tg(**vars(args), tb_writer=tb_writer, args=args)
    print(f"{time.time() - start_time:.4f} end time")


if __name__ == "__main__":
    main(get_args())


