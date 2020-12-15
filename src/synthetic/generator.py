import numpy as np
from collections import namedtuple

SyntheticSet = namedtuple("SyntheticSet", ("samples", "labels", "centers", "alpha_bet"))


def noise_a_set(s, alpha_bet):
    vec = np.zeros(shape=(alpha_bet,))
    for i in s:
        vec[i] = 1


def generate_synthetic_sets(num_samples,
                            num_centers,
                            alpha_bet,
                            noise,
                            min_num,
                            max_num,
                            random):
    # Generate the centers
    centers = list()
    vec = random.permutation(alpha_bet)
    for _ in range(num_centers):
        vec = random.permutation(vec)
        idx_final=random.choice(max_num - min_num) + min_num
        centers.append(vec[:idx_final].tolist())

    samples = list()
    labels = list()

    for _ in range(num_samples):
        label = random.randint(num_centers)
        center = centers[label]


    return SyntheticSet(samples=samples, centers=centers, alpha_bet=alpha_bet)
