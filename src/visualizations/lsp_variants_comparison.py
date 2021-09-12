import numpy as np
from clearml import Task
import matplotlib.pyplot as plt


def seperate_labels(x_values, y_values, labels, num_funcs):
    unique_funcs = np.unique(num_funcs)
    unique_labels = np.unique(labels)
    data = dict()
    for n_funcs in unique_funcs:
        data[n_funcs] = dict()
        for label in unique_labels:
            data[n_funcs][label] = {'x': [], 'y': []}

    for n_funcs, n_data in data.items():
        for pruning_method, p_data in n_data.items():
            idx = [i for i in range(len(num_funcs)) if num_funcs[i] == n_funcs and labels[i] == pruning_method]
            x_vals = [x_values[i] for i in idx]
            y_vals = [y_values[i] for i in idx]
            p_data['x'] = x_vals
            p_data['y'] = y_vals

    return data


def compare_clearml_project(seperate_legends, project_name=None, tasks=None, x_name=None, y_name=None):
    tasks = Task.get_tasks(project_name=project_name) if tasks is None else tasks

    x_y_values = []
    for task in tasks:
        try:
            x_y_values.append((float(task.data.hyperparams['Args'][x_name].value),
                               float(task.data.hyperparams['General'][y_name].value),
                               task.data.hyperparams['Args'][seperate_legends].value,
                               float(task.data.hyperparams['Args']['num_minhash_funcs'].value)))
        except Exception as e:
            print(e)
            pass

    x_values = [x for x, y, sep, sep2 in x_y_values]
    y_values = [y for x, y, sep, sep2 in x_y_values]
    labels1 = [sep for x, y, sep, sep2 in x_y_values]
    labels2 = [sep2 for x, y, sep, sep2 in x_y_values]
    return x_values, y_values, labels1, labels2


def get_pruning_results(project_name, pruning_methods, y_label):
    print("Retrieving tasks...")
    tasks = Task.get_tasks(project_name=project_name)
    print('Done')

    # logger = get_clearml_logger(project_name, 'Summary').logger

    print(f"Generating graph comparison for {y_label}")
    x_values, y_values, labels, num_funcs = compare_clearml_project(pruning_methods,
                                                         project_name=project_name,
                                                         tasks=tasks,
                                                         x_name='dim_edges',
                                                         y_name=y_label)
    data = seperate_labels(x_values, y_values, labels, num_funcs)
    for n_funcs, n_funcs_data in data.items():
        plt.figure()
        for method, m_data in n_funcs_data.items():
            x = m_data['x']
            y = m_data['y']
            order = np.argsort(x)
            x = [x[i] for i in order]
            y = [y[i] for i in order]
            plt.plot(x, y, label=method)

        plt.title(f"Prunning using {n_funcs} minhash functions")
        plt.xlabel("Edge dim")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    get_pruning_results(project_name=r'GNN_synthetic_pruning_dimensionality',
                        pruning_methods='pruning_method',
                        y_label='max test accuracy')
