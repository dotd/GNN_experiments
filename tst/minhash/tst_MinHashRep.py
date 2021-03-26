from src.utils.minhash_tools import MinHashRep

S, M = [None] * 4, [None] * 4
S[0], M[0] = [0, 3], ["_0", "_3"]
S[1], M[1] = [2], ["_2"]
S[2], M[2] = [1, 3, 4], ["_1", "_3", "_4"]
S[3], M[3] = [0, 2, 3], ["_0", "_2", "_3"]

minhash = MinHashRep(N=2, perms=[(1, 1), (3, 1)], prime=5, max_val=7, random=None)
for i in range(4):

    res = minhash.apply(S[i], M[i])
    print(res)
