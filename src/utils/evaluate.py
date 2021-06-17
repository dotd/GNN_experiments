import numpy as np

from ogb.graphproppred import Evaluator as OriginalEvaluator

try:
    import torch
except ImportError:
    torch = None


### Evaluator for graph classification


# Original file from OGB, Edit by Eitan Kosman, BCAI


class Evaluator(OriginalEvaluator):
    def __init__(self, name):
        # Initialize the parent object for an OGB dataset, because it recognizes only OGB datasets
        fake_name = name if name in ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2'] else 'ogbg-ppa'
        super(Evaluator, self).__init__(fake_name)

        """
        mnist, zinc, etc. are not support by the original evaluator. Here, I manually define the required
        parameters for using this evaluator for these datasets 
        """
        self.name = name
        self.num_tasks = 1

        if name in ['mnist']:
            # Add here datasets that use ACCURACY for evaluation
            self.eval_metric = 'acc'
        elif name in ['zinc', 'QM9']:
            # Add here datasets that use MAE for evaluation
            self.eval_metric = 'mae'
        else:
            raise NotImplementedError(f"Dataset {name} not implemented for Evaluator")

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric in ['rocauc', 'ap', 'rmse', 'acc', 'mae']:
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred

        elif self.eval_metric == 'F1':
            if not 'seq_ref' in input_dict:
                raise RuntimeError('Missing key of seq_ref')
            if not 'seq_pred' in input_dict:
                raise RuntimeError('Missing key of seq_pred')

            seq_ref, seq_pred = input_dict['seq_ref'], input_dict['seq_pred']

            if not isinstance(seq_ref, list):
                raise RuntimeError('seq_ref must be of type list')

            if not isinstance(seq_pred, list):
                raise RuntimeError('seq_pred must be of type list')

            if len(seq_ref) != len(seq_pred):
                raise RuntimeError('Length of seq_true and seq_pred should be the same')

            return seq_ref, seq_pred

        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def eval(self, input_dict):
        if self.eval_metric == 'mae':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_mae(y_true, y_pred)
        else:
            return super(Evaluator, self).eval(input_dict)

    def _eval_mae(self, y_true, y_pred):
        mae_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            mae_list.append(abs((y_true[is_labeled, i] - y_pred[is_labeled, i])).mean())

        return {'mae': sum(mae_list) / len(mae_list)}
