# Taken from
# https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation

from scipy.spatial.distance import cosine
from random import randint
import numpy as np

# specify the length of each minhash vector
N = 3
max_val = (2 ** 32) - 1

# create N tuples that will serve as permutation functions
# these permutation values are used to hash all input sets
perms = [(randint(0, max_val), randint(0, max_val)) for i in range(N)]

# initialize a sample minhash vector of length N
# each record will be represented by its own vec
vec = [float('inf') for i in range(N)]


def minhash(s, prime=4294967311):
    '''
    Given a set `s`, pass each member of the set through all permutation
    functions, and set the `ith` position of `vec` to the `ith` permutation
    function's output if that output is smaller than `vec[i]`.
    '''
    # initialize a minhash of length N with positive infinity values
    vec = [float('inf') for i in range(N)]

    for val in s:

        # ensure s is composed of integers
        if not isinstance(val, int): val = hash(val)

        # loop over each "permutation function"
        for perm_idx, perm_vals in enumerate(perms):
            a, b = perm_vals

            # pass `val` through the `ith` permutation function
            output = (a * val + b) % prime

            # conditionally update the `ith` value of vec
            if vec[perm_idx] > output:
                vec[perm_idx] = output

    # the returned vector represents the minimum hash of the set s
    return vec


import numpy as np

# specify some input sets
data1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets'])
data2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents'])

# get the minhash vectors for each input set
vec1 = minhash(data1)
vec2 = minhash(data2)

print(f"vec1={vec1}")
print(f"vec2={vec2}")

# divide both vectors by their max values to scale values {0:1}
vec1 = np.array(vec1) / max(vec1)
vec2 = np.array(vec2) / max(vec2)

# measure the similarity between the vectors using cosine similarity
print( ' * similarity:', 1 - cosine(vec1, vec2) )