import numpy as np

from src.utils.lsh_euclidean_tools import LSH, print_lsh_func
from src.synthetic.euclidean_generator import create_gauss_data
from src.utils.utils import dict_hierarchical


def tst_lsh_clustering():
    # Create the dataset
    din = 100
    num_centers = 5
    num_samples = 100
    noise = 1
    random = np.random.RandomState(0)

    euclidean_dataset = create_gauss_data(din=din,
                                          num_centers=num_centers,
                                          num_samples=num_samples,
                                          noise=noise,
                                          random=random)
    print(euclidean_dataset)

    num_functions = 1
    sparsity = 2
    std_of_threshold = 1
    random = np.random.RandomState(0)

    lsh = LSH(din,
              num_functions,
              sparsity,
              std_of_threshold,
              random)

    dict_label_signature = dict_hierarchical()
    dict_signature_label = dict_hierarchical()

    for i in range(len(euclidean_dataset.samples)):
        sample = euclidean_dataset.samples[i]
        label = euclidean_dataset.labels[i]
        signature = lsh.sign_vector(sample)
        dict_label_signature.inc([label, tuple(signature)])
        dict_signature_label.inc([tuple(signature), label])

    print(f"dict_label_signature\n{dict_label_signature}")
    print(f"dict_signature_label\n{dict_signature_label}")




if __name__ == "__main__":
    tst_lsh_clustering()
