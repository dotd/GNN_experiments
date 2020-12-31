import numpy as np

from src.utils.lsh_euclidean_tools import LSH, print_lsh_func


def tst_lsh_basic():
    din = 40
    num_functions = 2
    sparsity = 3
    std_of_threshold = 1
    random = np.random.RandomState(0)

    lsh = LSH(din,
              num_functions,
              sparsity,
              std_of_threshold,
              random)

    print(f"{print_lsh_func(lsh.lsh_funcs)}")
    vec = random.normal(size=(din,))
    signature = lsh.sign_vector(vec)
    print(f"vec={vec}")
    print(f"signature={signature}")


if __name__ == "__main__":
    tst_lsh_basic()
