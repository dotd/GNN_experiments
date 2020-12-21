import numpy as np


class LSH:

    def __init__(self,
                 din,
                 num_functions,
                 sparsity,
                 std_of_threshold,
                 random=np.random.RandomState()):
        self.din = din
        self.num_functions = num_functions
        self.sparsity = sparsity
        self.std_of_threshold = std_of_threshold
        self.random = random

        # Create the functions by the parameters
        self.indices = list()
        self.lsh_thresholds = list()
        for i in range(self.num_functions):
            self.indices.append(random.permutation(self.din)[0:self.sparsity])
            self.lsh_thresholds.append(self.random.normal(0, self.std_of_threshold, size=self.sparsity))

    def sign_vector(self, vec):
        signatures = list()
        for i in range(self.num_functions):
            signatures.append(vec[self.indices[i]] <= self.lsh_thresholds[i])
        return np.concatenate(signatures, axis=0) + 0


def print_lsh_func(mat):
    return "\n".join([",".join(["" if x == 0.0 else f"{x}" for x in mat[y, :]]) for y in range(mat.shape[0])])
