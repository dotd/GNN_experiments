import numpy as np


def create_gauss_data(din, num_centers, num_samples, random, noise):
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
    centers = random.normal(size=(din, num_centers))  # initialize centers
    samples = np.zeros(shape=(din, num_samples))  # initialize samples
    labels = np.zeros(shape=(num_samples,), dtype=int)  # initialize Labels

    # create the data
    for idx_sample in range(num_samples):
        # the class
        c = random.randint(num_centers)
        # put in labels
        labels[idx_sample] = int(c)
        # get the relevant center
        samples[:, idx_sample] = centers[:, c]
        for j in range(din):
            samples[j, idx_sample] += random.normal() * noise

    return centers, samples, labels
