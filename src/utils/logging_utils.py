import logging
import sys

import numpy as np
from clearml import Task
import matplotlib.pyplot as plt


def register_logger(log_file=None, stdout=True):
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def log_args_description(args):
    """
    Logs the content of the arguments
    Args:
        args: instance of arguments object, i.e. parser.parse_args()
    """
    args_header = """
    ====== All settings used ======:\n
"""

    s = ""
    for k, v in sorted(vars(args).items()):
        s += f"      {k}: {v}\n"

    logging.info(args_header + s)


def get_clearml_logger(project_name, task_name, tags=None):
    task = Task.init(project_name=project_name, task_name=task_name, tags=tags)
    logger = task.get_logger()
    return task


def log_command():
    logging.info("Running command:")
    logging.info(' '.join(sys.argv))


def compare_clearml_project(seperate_legends, project_name=None, tasks=None, x_name=None, y_name=None):
    tasks = Task.get_tasks(project_name=project_name) if tasks is None else tasks

    x_y_values = []
    for task in tasks:
        try:
            x_y_values.append((float(task.data.hyperparams['General'][x_name].value),
                               float(task.data.hyperparams['General'][y_name].value),
                               task.data.hyperparams['Args'][seperate_legends].value))
        except Exception as e:
            print(e)
            pass
    # x_y_values = [(float(task.data.hyperparams['General'][x_name].value),
    #                float(task.data.hyperparams['General'][y_name].value),
    #                task.data.hyperparams['Args'][seperate_legends].value) for task in tasks]

    x_values = [x for x, y, sep in x_y_values]
    y_values = [y for x, y, sep in x_y_values]
    labels = [sep for x, y, sep in x_y_values]
    return x_values, y_values, labels


def seperate_labels(x_values, y_values, labels):
    unique_labels = np.unique(labels)
    labeled_x = {l: [] for l in unique_labels}
    labeled_y = {l: [] for l in unique_labels}
    for x, y, label in zip(x_values, y_values, labels):
        labeled_x[label].append(x)
        labeled_y[label].append(y)

    return labeled_x, labeled_y, unique_labels


def summary_clearml_project(project_name, x_label, seperate_legends):
    print("Retrieving tasks...")
    tasks = Task.get_tasks(project_name=project_name)
    print('Done')
    y_labels = set(tasks[0].data.hyperparams['General'])
    y_labels.remove(x_label)

    logger = get_clearml_logger(project_name, 'Summary').logger
    for y_label in y_labels:
        print(f"Generating graph comparison for {y_label}")
        x_values, y_values, labels = compare_clearml_project(seperate_legends,
                                                             project_name=project_name,
                                                             tasks=tasks,
                                                             x_name=x_label,
                                                             y_name=y_label)
        labeled_x, labeled_y, unique_labels = seperate_labels(x_values, y_values, labels)
        for label in unique_labels:
            # plt.plot(labeled_x[label], labeled_y[label], label=label)
            scatter2d = np.vstack((labeled_x[label], labeled_y[label])).T
            logger.report_scatter2d(title=y_label, series=label, iteration=0, scatter=scatter2d,
                                    xaxis=x_label, yaxis=y_label)


def get_pruning_results(project_name, pruning_methods, y_label):
    print("Retrieving tasks...")
    tasks = Task.get_tasks(project_name=project_name)
    print('Done')

    # logger = get_clearml_logger(project_name, 'Summary').logger

    print(f"Generating graph comparison for {y_label}")
    x_values, y_values, labels = compare_clearml_project(pruning_methods,
                                                         project_name=project_name,
                                                         tasks=tasks,
                                                         x_name='keep edges',
                                                         y_name=y_label)
    labeled_x, labeled_y, unique_labels = seperate_labels(x_values, y_values, labels)
    for label in unique_labels:
        print(f'======================= {label} =======================')
        x = np.array(labeled_x[label])
        y = np.array(labeled_y[label])
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        for x_, y_ in zip(x, y):
            print(f'({x_}, {y_})')

        print()


if __name__ == '__main__':
    # summary_clearml_project(project_name=r'GNN_Cora_gat_sage',
    #                         x_label=r'keep edges',
    #                         seperate_legends='pruning_method')

    get_pruning_results(project_name=r'GNN_synthetic_pruning',
                        pruning_methods='pruning_method',
                        y_label='time/val')
