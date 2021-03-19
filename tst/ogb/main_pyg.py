import argparse
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torchvision import transforms

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from src.utils.proxy_utils import set_proxy
from tst.ogb.encoder_utils import augment_edge, decode_arr_to_seq, encode_y_to_arr, get_vocab_mapping
from tst.ogb.exp_utils import get_loss_function, evaluate, train
from tst.ogb.model_and_data_utils import add_zeros, create_model


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
                        help='dataset name (default: ogbg-molhiv)',
                        choices=['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2'])
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--proxy', action="store_true", default=False,
                        help="Set proxy env. variables. Need in bosch networks.", )
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
    print(f"DEBUG: Loaded the dataset: {dataset.name}")
    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    cls_criterion = get_loss_function(dataset.name)
    # The following is only used in the evaluation of the ogbg-code classifier.
    idx2word_mapper = None
    # specific transformations for the ogbg-code dataset
    if args.dataset in ['ogbg-code']:
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
        dataset.transform = transforms.Compose(
            [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
        idx2word_mapper = partial(decode_arr_to_seq, idx2vocab=idx2vocab)

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # original
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

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve)).item()
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print(f'Best validation score: {valid_curve[best_val_epoch]}')
    print(f'Test score: {test_curve[best_val_epoch]}')

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
