import numpy as np
from collections import namedtuple
from copy import deepcopy


class SyntheticSet:

    def __init__(self, samples=None,
                 labels=None,
                 centers=None,
                 alpha_bet=None):
        self.samples = samples
        self.labels = labels
        self.centers = centers
        self.alpha_bet = alpha_bet

    def __str__(self):
        s = list()
        s.append(f"samples={self.samples.__str__()}")
        s.append(f"labels={self.labels.__str__()}")
        s.append(f"centers={self.centers.__str__()}")
        s.append(f"alpha_bet={self.alpha_bet.__str__()}")
        return "\n".join(s)


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
        idx_final = random.choice(max_num - min_num) + min_num
        vec_sort = vec[:idx_final].tolist()
        vec_sort.sort()
        centers.append(vec_sort)

    samples = list()
    labels = list()

    for _ in range(num_samples):
        label = random.randint(num_centers)
        labels.append(label)
        s = centers[label]
        vec = np.zeros(shape=(alpha_bet,))
        vec[s] = 1
        noise_vec = random.choice(2, size=(alpha_bet,), p=(1 - noise, noise))
        vec = (vec + noise_vec)
        sample = np.where(vec == 1)[0].tolist()
        samples.append(sample)

    return SyntheticSet(samples=samples, labels=labels, centers=centers, alpha_bet=alpha_bet)
