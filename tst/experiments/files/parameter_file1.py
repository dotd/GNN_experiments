from dataclasses import dataclass

seed = 12345


@dataclass
class DatasetParams:
    num_samples = 100
    num_classes = 10
    min_nodes = 10
    max_nodes = 15
    noise_nodes = 0
    connectivity_rate = 0.8
    connectivity_rate_noise = 0.05
    noise_remove_node = 0.02
    noise_add_node = 0.02
    symmetric_flag = True


@dataclass
class CommonParams:
    dim_nodes = 10


@dataclass
class MinhashParams:
    num_minhash_funcs = 3


@dataclass
class LSHParams:
    lsh_num_funcs = 3
    sparsity = 4
    std_of_threshold = 1


@dataclass
class ModelParams:
    hidden_channels = 60
