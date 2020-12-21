import numpy as np
from scipy.spatial.distance import cosine
from src.utils.minhash_tools import MinHash
from src.utils.minhash_tools import create_jaacard_table, create_agreement_table


# specify some input sets
data1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets'])
data2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents'])

# get the minhash vectors for each input set
rnd = np.random.RandomState(0)
minhash = MinHash(N=3, rnd=rnd)
vec1 = minhash.apply(data1)[0]
vec2 = minhash.apply(data2)[0]

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
S = [None] * 4
S[0] = [0, 3]
S[1] = [2]
S[2] = [1, 3, 4]
S[3] = [0, 2, 3]

rnd = np.random.RandomState(0)
minhash = MinHash(N=2, perms=[(1, 1), (3, 1)], prime=5, max_val=7, rnd=rnd)
vecs = list()
vec_vals = list()
for i in range(4):
    vec, vec_val, translation = minhash.apply(S[i])
    print(f"i={i} vec={vec}")
    print(f"i={i} vec_val={vec_val}")
    print(f"i={i} translation={translation}")
    vecs.append(set(vec))
    vec_vals.append(set(vec_val))

print(f"vecs table=\n{create_jaacard_table(vecs)}")
print(f"vec_vals table=\n{create_jaacard_table(vec_vals)}")

# -------------------------------------------------
# This is from section 3.3.5, pp. 84--86 in the book: "Mining of Massive Datasets"
S = [None] * 4
S[0] = ["a", "d"]
S[1] = ["c"]
S[2] = ["b", "d", "e"]
S[3] = ["a", "c", "d"]

rnd = np.random.RandomState(0)
minhash = MinHash(N=3, perms=[(1, 1), (3, 1)], prime=57, max_val=57, rnd=rnd)
vecs = list()
vec_vals = list()
for i in range(4):
    vec, vec_val, translation = minhash.apply(S[i])
    print(f"i={i} vec={vec}")
    print(f"i={i} vec_val={vec_val}")
    print(f"i={i} translation={translation}")
    vecs.append(vec)
    vec_vals.append(vec_val)

print(f"vecs jaacard table=\n{create_jaacard_table(vecs)}")
print(f"vec_vals jaacard table=\n{create_jaacard_table(vec_vals)}")

print(f"vecs agreement table=\n{create_agreement_table(vecs)}")
print(f"vec_vals agreement table=\n{create_agreement_table(vec_vals)}")


