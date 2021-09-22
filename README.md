# LSP : Acceleration and Regularization of Graph Neural Networks via Locality Sensitive Pruning of Graphs

This repository contains the code of the experiments performed for our paper.

For instructions for how to run the code, please see the following.


## General instructions
Each script has a few input parameters passed via `argparse`. The parameters we use for pruning are:
* pruning_method - the pruning methodology used. `choices=["minhash_lsh_thresholding", "minhash_lsh_projection", "random"]`
* random_pruning_prob - relevant for random pruning only. indicates the amount of edges preserved from the original graph.
* num_minhash_funcs - relevant for minhash_lsh pruning methods only. indicates the number of hash functions used for the algorithm.
* sparsity - relevant for minhash_lsh pruning methods only. indicates the amount of entries used for calculating signatures.
* quantization_step - relevant for `minhash_lsh_projection` only. indicates the bin length for hashing with random projections.


## Running node classification experiments
This section of experiments divides into 2 parts:
* Running experiments on the PPI dataset
* Running experiments on all other datasets

### Running node classification  experiments on PPI
These experiments are run via the script called `tst.ogb.main_with_pruning_node_prediction_on_ppi.py`. for example, we run an experiments via executing the following command from the root directory:

```commandline
python -m tst.ogb.main_with_pruning_node_prediction_on_ppi --pruning_method random --random_pruning_prob 0.5 --device cuda:0
```

This will execute an experiment of training and testing with pruning the PPI graphs with `random` and `p = 0.5`.

### Running node classification experiments on other datasets
These experiments are run via the script called `tst.ogb.main_with_pruning_node_prediction.py`. for example, we run an experiments via executing the following command from the root directory:

```commandline
python -m tst.ogb.main_with_pruning_node_prediction_on_ppi --dataset CiteSeer --gnn sage --pruning_method random --random_pruning_prob 0.5  --device 0
```

This will execute an experiment of training and testing with pruning the CiteSeer graphs with `random` and `p = 0.5`. The experiment would be executed on The first cuda device, i.e. `cuda:0`. the graph neural network that would be used for this experiment would be `GraphSage` as described in the paper.
Possible parameters:
* dataset - `choices=['github', 'ogbg-molhiv', "Cora", "CiteSeer", "PubMed"]`
* gnn - `choices=['sage', 'gat']`


## Running graph regression experiments


## Running graph classification experiments on the synthetic data