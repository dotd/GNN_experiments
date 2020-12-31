import numpy as np

# Taken from
# https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation


class MinHash:

    def __init__(self,
                 N,  # specify the length of each minhash vector
                 random,
                 max_val=(2 ** 32) - 1,
                 perms=None,
                 prime=4294967311
                 ):
        self.N = N
        self.max_val = max_val
        self.prime = prime
        if perms is None:
            self.perms = [(random.randint(0, min(self.max_val, prime)), random.randint(0, min(self.max_val, prime))) for i in range(self.N)]
        else:
            self.perms = perms

    def apply(self, s):
        vec = [float('inf') for i in range(self.N)]
        vec_val = [None for i in range(self.N)]
        translation = dict()

        for val_orig in s:

            # ensure s is composed of integers
            if not isinstance(val_orig, int):
                val = hash(val_orig) % self.prime
            else:
                val = val_orig
            translation[val] = val_orig

            # loop over each "permutation function"
            for perm_idx, perm_vals in enumerate(self.perms):
                a, b = perm_vals

                # pass `val` through the `ith` permutation function
                output = (a * val + b) % self.prime

                # conditionally update the `ith` value of vec
                if vec[perm_idx] > output:
                    vec[perm_idx] = output
                    vec_val[perm_idx] = val_orig

        # the returned vector represents the minimum hash of the set s
        return vec, vec_val, translation

    def __str__(self):
        s = list()
        s.append(f"N={self.N}")
        s.append(f"max_val={self.max_val}")
        s.append(f"prime={self.prime}")
        for i, perm in enumerate(self.perms):
            s.append(f"func ({i}={self.perms[i][0]} * val + {self.perms[i][1]}) % {self.prime}")
        return "\n".join(s)

def compute_jaacard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
    return actual_jaccard


def compute_agreement(s1, s2):
    return np.sum([1 if x==y else 0 for (x, y) in zip(s1,s2)]) / len(s1)


def create_jaacard_table(sets):
    results = list()
    for i1 in range(0, len(sets) - 1):
        for i2 in range(i1+1, len(sets)):
            results.append([i1, i2, compute_jaacard(sets[i1], sets[i2])])
    return results


def create_agreement_table(sets):
    results = list()
    for i1 in range(0, len(sets) - 1):
        for i2 in range(i1+1, len(sets)):
            results.append([i1, i2, compute_agreement(sets[i1], sets[i2])])
    return results
