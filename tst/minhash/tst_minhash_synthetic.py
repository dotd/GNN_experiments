import numpy as np
from src.utils.minhash_tools import MinHash
from src.synthetic.set_generator import generate_synthetic_sets
from src.utils.minhash_tools import create_jaacard_table, create_agreement_table
import json
import pprint

# ---------------------------------------------------------
# we generate classification dataset and test it.

rnd = np.random.RandomState(0)
minhash = MinHash(N=2,
                  rnd=rnd)

synthetic_set = generate_synthetic_sets(num_samples=100,
                                        num_centers=4,
                                        alpha_bet=20,
                                        noise=0.1,
                                        min_num=6,
                                        max_num=8,
                                        random=rnd)

results = list()
for i in range(len(synthetic_set.samples)):
    sample = synthetic_set.samples[i]
    label = synthetic_set.labels[i]
    vec, vec_val, translation = minhash.apply(sample)
    results.append([tuple(sample), label, tuple(vec_val)])

# Sort raws by label
results.sort(key=lambda x: x[1])
for result in results:
    print(f"sample={result[0]} \tlabel={result[1]} \tvec_val={result[2]}")


stats_by_vec_val = dict()
stats_by_label = dict()
for result in results:
    vec_val = result[2]
    label = result[1]
    if vec_val not in stats_by_vec_val:
        stats_by_vec_val[vec_val] = dict()

    if label not in stats_by_vec_val[vec_val]:
        stats_by_vec_val[vec_val][label] = 0
    stats_by_vec_val[vec_val][label] += 1

    if label not in stats_by_label:
        stats_by_label[label] = dict()

    if vec_val not in stats_by_label[label]:
        stats_by_label[label][vec_val] = 0
    stats_by_label[label][vec_val] += 1

print(stats_by_vec_val)
print(stats_by_label)
pp = pprint.PrettyPrinter(depth=4)
pp.pprint(stats_by_label)


# Analysis of errors per label
