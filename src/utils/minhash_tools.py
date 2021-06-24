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


class MH:

    def __init__(self, value=float('inf'), obj=None, meta=None):
        self.value = value
        self.obj = obj
        self.meta = meta

    def __str__(self):
        return f"(value={self.value}, obj={self.obj}, meta={self.meta})"

    def __repr__(self):
        return self.__str__()


class MinHashRep:

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

    def apply(self, s, metas=None):
        result = []
        all_minhashes = []
        if metas is None:
            metas = s
        for idx, perm in enumerate(self.perms):
            minimal_value = MH()
            for val_orig, meta in zip(s, metas):
                # ensure s is composed of integers
                if not isinstance(val_orig, int):
                    val = (hash(val_orig) % self.prime, val_orig)
                else:
                    val = (val_orig, val_orig)

                a, b = perm
                new_val = (a * val[0] + b) % self.prime
                if new_val < minimal_value.value:
                    minimal_value = MH(new_val, val[1], meta)
            result.append(minimal_value)


        # the returned vector represents the minimum hash of the set s
        # result = list(set(s) - set(result))

        return result

    def __str__(self):
        s = list()
        s.append(f"N={self.N}")
        s.append(f"max_val={self.max_val}")
        s.append(f"prime={self.prime}")
        for i, perm in enumerate(self.perms):
            s.append(f"func ({i}={self.perms[i][0]} * val + {self.perms[i][1]}) % {self.prime}")
        return "\n".join(s)
