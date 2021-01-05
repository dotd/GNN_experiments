"""
In this tst we show how to have several parameters at once.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.utils.utils import product_dict
from src.experiments.single_runner_type02 import single_runner_type02


def tst_single_runner_type02():
    params = {
        "seed": [0, 12345],
        "dataset_params.num_samples": [100],
        "dataset_params.num_classes": [10],
        "dataset_params.min_nodes": [10],
        "dataset_params.max_nodes": [15],
        "common_params.dim_nodes": [11],
        "dataset_params.connectivity_rate": [0.2, 0.3, 0.4],
        "dataset_params.connectivity_rate_noise": [0.05],
        "dataset_params.noise_nodes": [1],
        "dataset_params.noise_remove_node": [0.01],
        "dataset_params.noise_add_node": [0.01],
        "dataset_params.symmetric_flag": [True],
        #
        "minhash_params.num_minhash_funcs": [3],
        #
        "lsh_params.lsh_num_funcs": [3],
        "lsh_params.sparsity": [4],
        "lsh_params.std_of_threshold": [1],
        "model_params.hidden_channels": [60],
        "model_params.num_episodes": [5]
    }

    parameter_settings = list(product_dict(**params))
    with ProcessPoolExecutor(max_workers=3) as executor:
        res = executor.map(single_runner_type02, parameter_settings)

    print(list(res))


if __name__ == "__main__":
    tst_single_runner_type02()
