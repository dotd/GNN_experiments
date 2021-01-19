import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from src.utils.proxy_utils import set_proxy
from tst.ogb.encoder_utils import ASTNodeEncoder, augment_edge, decode_arr_to_seq, encode_y_to_arr, get_vocab_mapping
from tst.ogb.gcn import GCN


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


def create_model(dataset: PygGraphPropPredDataset, emb_dim: int, dropout_ratio: float, device: str, num_layers: int,
                 max_seq_len: int, num_vocab: int):
    print("creating a model for ", dataset.name)
    if dataset.name == "ogbg-molhiv":
        node_encoder = AtomEncoder(emb_dim=emb_dim)
        edge_encoder_constrtuctor = BondEncoder
        print("Number of classes: ", dataset.num_tasks)
        model = GCN(num_classes=dataset.num_tasks, num_layer=num_layers,
                    emb_dim=emb_dim, drop_ratio=dropout_ratio,
                    node_encoder=node_encoder, edge_encoder_ctor=edge_encoder_constrtuctor).to(device)

    elif dataset.name == "ogbg-code":
        nodetypes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'typeidx2type.csv.gz')
        nodeattributes_mapping = pd.read_csv(Path(dataset.root) / 'mapping' / 'attridx2attr.csv.gz')
        node_encoder = ASTNodeEncoder(emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                      num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)
        split_idx = dataset.get_idx_split()
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
        edge_encoder_ctor = partial(torch.nn.Linear, 2)
        print(f"Multiclassification with {len(vocab2idx)} classes. Num labels per example: {max_seq_len}")
        model = GCN(num_classes=len(vocab2idx), max_seq_len=max_seq_len, node_encoder=node_encoder,
                    edge_encoder_ctor=edge_encoder_ctor, num_layer=num_layers, emb_dim=emb_dim,
                    drop_ratio=dropout_ratio).to(device)

    else:
        raise ValueError("Used an invalid dataset name")
    return model


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
                        help='dataset name (default: ogbg-molhiv)')
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
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    cls_criterion = torch.nn.BCEWithLogitsLoss()
    # The following is only used in the evaluation of the ogbg-code classifier.
    idx2word_mapper = None
    # specific transformations for the ogbg-code dataset
    if args.dataset == 'ogbg-code':
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
        dataset.transform = transforms.Compose(
            [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])
        cls_criterion = torch.nn.CrossEntropyLoss()
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
