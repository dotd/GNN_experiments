import numpy as np
from profilehooks import timecall


class LSH:

    def __init__(self,
                 din,
                 num_functions,
                 sparsity,
                 std_of_threshold,
                 random,
                 **kwargs):
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
        # signatures = list()
        # for i in range(self.num_functions):
        #     signatures.append(vec[self.indices[i]] <= self.lsh_thresholds[i])
        signatures = [vec[self.indices[i]] <= self.lsh_thresholds[i] for i in range(self.num_functions)]
        return np.concatenate(signatures, axis=0).astype(int)

    def sign_vectors(self, vecs):
        try:
            signatures = np.array([vecs[:, self.indices[i]] <= self.lsh_thresholds[i] for i in range(self.num_functions)])
        except Exception as e:
            pass
        # return np.concatenate(signatures, axis=0).astype(int)
        return signatures.transpose((1, 0, 2)).reshape(signatures.shape[1], -1).astype(int)

    def __str__(self):
        s = list()
        s.append(f"din={self.din}")
        s.append(f"num_functions={self.num_functions}")
        s.append(f"sparsity={self.sparsity}")
        s.append(f"std_of_threshold={self.std_of_threshold}")
        for i in range(self.num_functions):
            s.append(f"func {i}, indices:{self.indices[i]} thresholds:{self.lsh_thresholds[i]}")
        return "\n".join(s)



def print_lsh_func(mat):
    return "\n".join([",".join(["" if x == 0.0 else f"{x}" for x in mat[y, :]]) for y in range(mat.shape[0])])
