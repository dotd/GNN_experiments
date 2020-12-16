from random import randint


# Taken from
# https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation


class MinHash:

    def __init__(self,
                 N=3,  # specify the length of each minhash vector
                 max_val=(2 ** 32) - 1,
                 perms=None,
                 prime=4294967311):
        self.N = N
        self.max_val = max_val
        self.prime = prime
        if perms is None:
            self.perms = [(randint(0, self.max_val), randint(0, self.max_val)) for i in range(self.N)]
        else:
            self.perms = perms

    def apply(self, s):
        vec = [float('inf') for i in range(self.N)]

        for val_orig in s:

            # ensure s is composed of integers
            if not isinstance(val_orig, int):
                val = hash(val_orig)
            else:
                val = val_orig

            # loop over each "permutation function"
            for perm_idx, perm_vals in enumerate(self.perms):
                a, b = perm_vals

                # pass `val` through the `ith` permutation function
                output = (a * val + b) % self.prime

                # conditionally update the `ith` value of vec
                if vec[perm_idx] > output:
                    vec[perm_idx] = output

        # the returned vector represents the minimum hash of the set s
        return vec
