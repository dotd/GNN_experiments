import torch
from tqdm import tqdm


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

            # ogbg-code is a multi-labelling task, so it needs to be treated diffrerently
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
            elif dataset_name == 'ogbg-ppa':
                y_true.append(batch.y.view(-1, 1).detach().cpu())
                y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
            else:
                raise AttributeError("Batch does not contain either a y-member or a y_arr-member")

    if dataset_name == 'ogbg-code':
        input_dict = {"seq_ref": y_true, "seq_pred": y_pred}
    elif dataset_name in ['ogbg-molhiv', 'ogbg-ppa']:
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
