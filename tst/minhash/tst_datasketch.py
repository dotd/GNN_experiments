from datasketch import MinHash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
         'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
         'estimating', 'the', 'similarity', 'between', 'documents']

num_perm = 3
m1 = MinHash(num_perm=num_perm)
m2 = MinHash(num_perm=num_perm)
for d in data1:
    m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))

print(f"m1={m1.permutations}")
print(f"m2={m2.permutations}")

s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
print("Actual Jaccard for data1 and data2 is", actual_jaccard)
print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))
