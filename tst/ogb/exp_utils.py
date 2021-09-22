import time

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(model, dataset, device, loader, optimizer, cls_criterion, tb_writer=None):
    model.train()

    start_time = time.time()
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
                if dataset.name in ['ogbg-ppa', 'mnist']:
                    loss = cls_criterion(pred.to(torch.float32),
                                         batch.y.view(-1, ))
                elif dataset.name in ['zinc', 'QM9']:
                    loss = cls_criterion(pred.flatten(), batch.y.flatten())
                elif dataset.name in ['ogbg-molhiv', 'ogbg-molpcba']:
                    # ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                raise AttributeError("Batch does not contain either a y-member or a y_arr-member")

            loss.backward()
            optimizer.step()

            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_iterations', loss.item(), tb_writer.iteration)
                tb_writer.iteration += 1

    end_time = time.time()

    seconds_per_iter = (end_time - start_time) / len(loader)

    return seconds_per_iter


def evaluate(model, device, loader, evaluator, arr_to_seq, dataset_name: str, return_avg_time: bool = False):
    if len(loader) == 0:
        return None

    model.eval()
    y_true = []
    y_pred = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                # ogbg-code is a multi-labelling task, so it needs to be treated differently
                if dataset_name == 'ogbg-code2':
                    mat = []
                    for i in range(len(pred)):
                        mat.append(torch.argmax(pred[i], dim=1).view(-1, 1))

                    mat = torch.cat(mat, dim=1)

                    seq_pred = [arr_to_seq(arr) for arr in mat]

                    seq_ref = [batch.y[i] for i in range(len(batch.y))]

                    y_true.extend(seq_ref)
                    y_pred.extend(seq_pred)
                elif dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred.append(pred.detach().cpu())
                elif dataset_name in ['ogbg-ppa', 'mnist']:
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                elif dataset_name in ['zinc', 'QM9']:
                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred.append(pred.detach().cpu())
                else:
                    raise AttributeError("Batch does not contain either a y-member or a y_arr-member")

    end_time = time.time()

    seconds_per_iter = (end_time - start_time) / len(loader)

    if dataset_name == 'ogbg-code2':
        input_dict = {"seq_ref": y_true, "seq_pred": y_pred}
    elif dataset_name in ['ogbg-molhiv', 'ogbg-ppa', 'ogbg-molpcba', 'mnist', 'zinc', 'QM9']:
        y_true = torch.cat(y_true, dim=0).numpy().reshape(-1, 1)
        y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1, 1)
        input_dict = {"y_true": y_true, "y_pred": y_pred}

    if return_avg_time:
        return evaluator.eval(input_dict), seconds_per_iter
    else:
        return evaluator.eval(input_dict)


def get_loss_function(dataset_name: str):
    loss = None
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        loss = torch.nn.BCEWithLogitsLoss()
    elif dataset_name in ['zinc', 'QM9']:
        loss = torch.nn.L1Loss()
    elif dataset_name in ['ogbg-code2', 'ogbg-ppa', 'mnist', 'cora', 'reddit']:
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("No loss function specified for the given database!")
    return loss
