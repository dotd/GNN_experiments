import numpy as np


class EuclideanDataset:

    def __init__(self, samples, labels, centers):
        self.samples = samples
        self.centers = centers
        self.labels = labels

    def __str__(self):
        s = list()
        s.append(f"centers:\n")
        for c in range(len(self.centers)):
            s.append(f"center {c:03}: {' '.join([f'{num:+2.4f}' for num in self.centers[c]])}\n")
        s.append(f"samples:\n")
        for i in range(len(self.samples)):
            s.append(f"sample {i:03}: {' '.join([f'{num:+2.4f}' for num in self.samples[i]])}\n")
        return "".join(s)


def create_gauss_data(din, num_centers, num_samples, noise, random):
    """
    This function generates gaussian data
    :param din: input dimension
    :param num_centers: output dimension
    :param num_samples: number of samples
    :param random: For reproducibility
    :param noise:
    :return:
    C: gaussian input distribution
    S: samples from space C
    L: samples labels
    """
    centers = random.normal(size=(num_centers, din))  # initialize centers
    samples = np.zeros(shape=(num_samples, din))  # initialize samples
    labels = np.zeros(shape=(num_samples,), dtype=int)  # initialize Labels

    # create the data
    for idx_sample in range(num_samples):
        # the class
        c = random.randint(num_centers)
        # put in labels
        labels[idx_sample] = int(c)
        # get the relevant center
        samples[idx_sample, :] = centers[c, :]
        for j in range(din):
            samples[idx_sample, j] += random.normal() * noise

    return EuclideanDataset(samples, labels, centers)
