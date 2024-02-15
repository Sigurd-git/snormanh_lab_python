import numpy as np
from functools import lru_cache


def subdivide(N, n_divisions):
    """
    Returns indices that subdivide a vector of length N into sections of
    approximately equal length (exactly equal if n_divisions exactly subdivides N).

    Example
    -------
    subdivide(10, 4)

    Created on 2016-06-25 by Sam NH
    """

    # size of each fold
    fold_sizes = np.full(n_divisions, N // n_divisions)
    fold_sizes[: N % n_divisions] += 1
    assert sum(fold_sizes) == N

    # indices for each fold
    inds = np.empty(N, dtype=int)
    for i in range(n_divisions):
        xi = np.arange(fold_sizes[i]) + sum(fold_sizes[:i])
        inds[xi] = (
            i + 1
        )  # note: "+1" is added because Python indices start at 0, but MATLAB indices start at 1

    return inds


@lru_cache(maxsize=None)
def partition_within_labels_v2(labels, partition_fractions, seed=None, shuffle=True):
    """
    Groups elements of an array into partitions within each label category separately,
    useful for evenly distributing examples per class in training/testing sets.

    Parameters:
    - labels (array-like of shape (n,)): Labels for each element, a one-dimensional array.
    - partition_fractions (array-like of shape (m,)): Fractions to split each label group into, a one-dimensional array. You can input a scalar(n_folds) and we will turn it into 1/n_folds later.
    - seed (int, optional): Seed for the random number generator for reproducible shuffling.
    - shuffle (bool): Whether to shuffle elements within each label group before partitioning.

    Returns:
    - partition_index (array of shape (n,)): A one-dimensional array indicating the partition index for each element.

    Each input and output is a one-dimensional array, reflecting the simplicity and specificity of the function's operation.
    """
    labels = np.array(labels)
    # Optionally set the random seed for fixed random decisions
    if seed is not None:
        np.random.seed(seed)

    # Turn integer into probability vector if needed
    if np.isscalar(partition_fractions):
        partition_fractions = np.ones(partition_fractions) / partition_fractions

    # Sort unique labels by frequency
    unique_labels, n_instances_per_label = np.unique(labels, return_counts=True)
    unique_labels = unique_labels[np.argsort(n_instances_per_label)]

    partition_index = np.full(len(labels), np.nan)
    n_partitions = len(partition_fractions)

    for label in unique_labels:
        # Indices for this label
        xi = np.where(labels == label)[0]

        # Shuffle the indices if requested
        if shuffle:
            np.random.shuffle(xi)

        # Calculate the number of samples in each partition and adjust for rounding
        n_samples_per_partition = np.floor(len(xi) * partition_fractions).astype(int)
        n_samples_to_add = len(xi) - n_samples_per_partition.sum()
        for i in range(n_samples_to_add):
            n_samples_per_partition[i % n_partitions] += 1

        # Assign samples to partitions
        start_idx = 0
        for i, n_samples in enumerate(n_samples_per_partition):
            partition_index[xi[start_idx : start_idx + n_samples]] = i + 1
            start_idx += n_samples
    partition_index = partition_index.astype(int)
    return partition_index


if __name__ == "__main__":
    print(
        subdivide(15, 3)
    )  # This should return array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.])

    print(subdivide(7, 2))  # This should return array([1., 1., 1., 1., 2., 2., 2.])
