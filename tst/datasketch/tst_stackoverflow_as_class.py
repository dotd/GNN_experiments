from scipy.spatial.distance import cosine
from src.tools.minhash_tools import MinHash

import numpy as np

# specify some input sets
data1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets'])
data2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents'])

# get the minhash vectors for each input set
minhash = MinHash(N=3)
vec1 = minhash.apply(data1)
vec2 = minhash.apply(data2)

print(f"vec1={vec1}")
print(f"vec2={vec2}")

# divide both vectors by their max values to scale values {0:1}
vec1 = np.array(vec1) / max(vec1)
vec2 = np.array(vec2) / max(vec2)

print(f"vec1={vec1}")
print(f"vec2={vec2}")

# measure the similarity between the vectors using cosine similarity
print(' * similarity:', 1 - cosine(vec1, vec2))

# -------------------------------------------------
# This is from section 3.3.5, pp. 84--86 in the book: "Mining of Massive Datasets"
S = [0] * 4
S[0] = [0, 3]
S[1] = [2]
S[2] = [1, 3, 4]
S[3] = [0, 2, 3]

minhash = MinHash(N=2, perms=[(1, 1), (3, 1)], prime=5, max_val=5)
for i in range(4):
    vec = minhash.apply(S[i])
    print(f"i={i} vec={vec}")
