import numpy as np

from src.utils.lsh_euclidean_tools import LSH, print_lsh_func
from src.synthetic.euclidean_generator import create_gauss_data
from src.utils.utils import pretty_vec


def tst_lsh_clustering():
    # Create the dataset
    din = 20
    num_centers = 5
    num_samples = 100
    noise = 0.1
    random = np.random.RandomState(0)

    euclidean_dataset = create_gauss_data(din=din,
                                          num_centers=num_centers,
                                          num_samples=num_samples,
                                          noise=noise,
                                          random=random)
    print(euclidean_dataset)

    num_functions = 2
    sparsity = 3
    std_of_threshold = 1
    random = np.random.RandomState(0)

    lsh = LSH(din,
              num_functions,
              sparsity,
              std_of_threshold,
              random)

    vec = random.normal(size=(din,))
    signature = lsh.sign_vector(vec)
    print(f"vec={pretty_vec(vec)}")
    print(f"signature={signature}")


if __name__ == "__main__":
    tst_lsh_clustering()
