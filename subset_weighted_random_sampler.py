from torch.utils.data import Sampler, WeightedRandomSampler
import numpy as np


class SubsetWeightedRandomSampler(Sampler):
    ''' A sampler that takes a subset of samples'''

    def __init__(self, indices, weights, num_samples, replacement=True):
        self.indices = indices
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.weighted_random = WeightedRandomSampler(
                weights, num_samples, replacement=replacement)
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(WeightedRandomSampler(
                self.weights, self.num_samples, replacement=self.replacement))
        return self

    def __next__(self):
        i = next(self.iterator)
        if i:
            return self.indices[i]
        else:
            return i

    def __len__(self):
        return self.num_samples


def compute_weights(labels, indices=None):
    labels = np.array(labels)
    if indices is not None:
        labels = labels[indices]
    counts = np.bincount(labels)
    labels_weights = 1. / counts
    weights = labels_weights[labels]
    return weights
