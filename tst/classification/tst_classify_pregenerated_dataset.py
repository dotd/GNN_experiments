import pickle
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.classification.tst_classify_compare_random_and_lsh_synthetic_tg import tst_classify_networkx_synthetic_tg, \
    get_args

start_time = time.time()

import src.synthetic.random_graph_dataset as rgd


def main(args):
    if args.generate_only:
        graph_dataset = rgd.generate_graphs_dataset(**vars(args), random=np.random.RandomState(0))
        with open(args.dataset_path, 'wb') as fp:
            pickle.dump(graph_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(args.dataset_path, 'rb') as fp:
            graph_dataset = pickle.load(fp)

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
        clearml_task = get_clearml_logger(project_name="GNN_synthetic_pruning_dimensionality",
                                          task_name=get_time_str(),
                                          tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    graph_dataset, prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = \
        tst_classify_networkx_synthetic_tg(**vars(args), tb_writer=tb_writer, args=args, graph_dataset=graph_dataset)
    print(f"{time.time() - start_time:.4f} end time")

    experiment_logs = dict()
    experiment_logs = clearml_task.connect(experiment_logs)
    experiment_logs['time/train'] = avg_time_train
    experiment_logs['time/val'] = avg_time_test
    experiment_logs['keep edges'] = prunning_ratio
    experiment_logs['max train accuracy'] = best_train
    experiment_logs['max test accuracy'] = best_test


if __name__ == "__main__":
    main(get_args())
