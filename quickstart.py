"""Train native NumPy e-means and assign independent new vectors on CPU."""

import argparse

import faiss
from faiss.contrib.e_means import NumpyEmeans
import numpy as np


def run(samples=200_000, new_samples=100_000, dimension=64, clusters=500, seed=7):
    for name, value in (("samples", samples), ("new_samples", new_samples),
                        ("dimension", dimension), ("clusters", clusters)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if clusters > samples:
        raise ValueError("clusters must be at most samples")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    rng = np.random.default_rng(seed)
    x = rng.random((samples, dimension), dtype="float32")
    em = NumpyEmeans(dimension, clusters, seed=seed)
    centroids, labels = em.train(x)
    x_new = rng.random((new_samples, dimension), dtype="float32")
    distances, new_labels = em.assign(x_new)
    return centroids, labels, distances, new_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--new-samples", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    faiss.omp_set_num_threads(args.threads)
    try:
        arrays = run(**{key: value for key, value in vars(args).items() if key != "threads"})
    except ValueError as error:
        parser.error(str(error))
    for name, array in zip(("centroids", "last-pass labels", "new squared distances", "new labels"), arrays):
        print(f"{name}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
