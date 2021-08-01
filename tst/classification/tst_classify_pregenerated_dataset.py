import argparse
import pickle

import numpy as np
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv, GCNConv

from src.archs.gat_geometric import GAT
from src.utils.csv_utils import prepare_csv
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.classification.tst_classify_compare_random_and_lsh_synthetic_tg import tst_classify_networkx_synthetic_tg, \
    get_args
from tst.ogb.main_pyg_with_pruning import prune_datasets, prune_dataset

start_time = time.time()

import src.synthetic.random_graph_dataset as rgd
from tst.torch_geometric.tst_torch_geometric1 import GCN
import src.synthetic.synthetic_utils as su
from tst.torch_geometric.tst_torch_geometric1 import train, func_test
from torch_geometric.data import DataLoader


@prepare_csv
def main(args):
    vals = dict()
    csv_file = args.csv_file

    if args.generate_only:
        graph_dataset = rgd.generate_graphs_dataset(**vars(args), random=np.random.RandomState(0))
        with open(args.dataset_path, 'wb') as fp:
            pickle.dump(graph_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

        exit()
    else:
        with open(args.dataset_path, 'rb') as fp:
            graph_dataset = pickle.load(fp)

    tb_writer = None
    if args.enable_clearml_logger:
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
        tst_classify_networkx_synthetic_tg(**vars(args), tb_writer=tb_writer, args=args, graph_dataset=graph_dataset)
    print(f"{time.time() - start_time:.4f} end time")

    vals['keep edges'] = prunning_ratio
    vals['train acc'] = best_train
    vals['test acc'] = best_test

    vals['train time'] = avg_time_train
    vals['test time'] = avg_time_test

    vals['architecture'] = args.gnn
    vals['pruning method'] = args.pruning_method
    df = pd.read_csv(csv_file)
    df = df.append(vals, ignore_index=True)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    main(get_args())
