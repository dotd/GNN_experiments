import time

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import NeighborSampler

from src.utils.csv_utils import prepare_csv
from src.utils.date_utils import get_time_str
from src.utils.logging_utils import get_clearml_logger
from tst.ogb.main_pyg_with_pruning import prune_dataset, get_args
from tst.ogb.main_with_pruning_node_prediction import get_model, get_dataset, test

start_time = time.time()
# Created by: Eitan Kosman, BCAI

sage = False


def run(args):
    dataset = get_dataset(args.dataset)
    data = dataset.data
    tb_writer = SummaryWriter()
    tb_writer.iteration = 0

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and args.device != 'cpu' else torch.device("cpu")
    model = get_model(dataset.data.num_features, dataset.num_classes, args.gnn)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    old_edge_count = data.edge_index.shape[1]

    # Pass the whole graph to the pruning mechanism. Consider it as one sample
    pruning_params, prunning_ratio = prune_dataset([data], args, random=np.random.RandomState(0), pruning_params=None)

    edge_count = data.edge_index.shape[1]
    print(
        f"Old number of edges: {old_edge_count}. New one: {edge_count}. Change: {(old_edge_count - edge_count) / old_edge_count * 100}\%")

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=1024, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)

    best_train = 0
    best_test = 0
    train_times = []
    test_times = []
    for epoch in range(1, args.epochs + 1):
        loss, acc, f1, avg_time_train = train(epoch, dataset, train_loader, model, device, optimizer, tb_writer)
        train_times.append(avg_time_train)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {f1:.4f}')

        train_acc, val_acc, test_acc, avg_time_test = test(dataset, subgraph_loader, model, device)
        test_times.append(avg_time_test)
        print(f'Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}, '
              f'Test ACC: {test_acc:.4f}')

        tb_writer.add_scalars('Accuracy',
                              {'train': train_acc,
                               'Validation': val_acc,
                               'Test': test_acc},
                              epoch)

        best_train = max(best_train, train_acc)
        best_test = max(best_test, test_acc)

    return prunning_ratio, best_train, best_test, avg_time_train, avg_time_test


@prepare_csv
def main(args, csv_file):
    vals = dict()

    """
    Pruning with LSH
    """
    args.pruning_method = 'minhash_lsh'
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
        clearml_logger = get_clearml_logger(project_name=f"GNN_{args.dataset}_{args.gnn}_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = run(args)
    print(f"{time.time() - start_time:.4f} end time")

    vals['keep edges'] = prunning_ratio
    vals['minhash train'] = best_train
    vals['minhash test'] = best_test

    vals['minhash time train'] = avg_time_train
    vals['minhash time test'] = avg_time_test

    print("Sleeping between experiments...")
    time.sleep(5)
    print("Wake up!")
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
        pruning_param_name = 'num_minhash_funcs' if args.pruning_method == 'minhash_lsh' else 'random_pruning_prob'
        pruning_param = args.num_minhash_funcs if args.pruning_method == 'minhash_lsh' else args.random_pruning_prob
        tags.append(f'{pruning_param_name}: {pruning_param}')
        clearml_logger = get_clearml_logger(project_name=f"GNN_{args.dataset}_{args.gnn}_pruning",
                                            task_name=get_time_str(),
                                            tags=tags)

    print(f"{time.time() - start_time:.4f} start time")
    prunning_ratio, best_train, best_test, avg_time_train, avg_time_test = run(args)
    print(f"{time.time() - start_time:.4f} end time")
    vals['random train'] = best_train
    vals['random test'] = best_test
    vals['random time train'] = avg_time_train
    vals['random time test'] = avg_time_test
    df = pd.read_csv(csv_file)
    df = df.append(vals, ignore_index=True)
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    args = get_args()
    args.csv_file = f"{args.dataset}_{args.gnn}_results.csv"
    main(args)
