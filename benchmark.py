"""Compare native CPU e-means and Lloyd clustering on a shared dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from time import perf_counter

import faiss
from faiss.contrib.e_means import NumpyEmeans
import numpy as np

from report import render_report


@dataclass(frozen=True)
class Config:
    samples: int = 6000
    dimension: int = 8
    clusters: int = 8
    epochs: float = 3.0
    lloyd_epochs: int = 25
    seed: int = 7
    threads: int = 1

    def validate(self) -> None:
        for name in ("samples", "dimension", "clusters", "lloyd_epochs", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.samples < self.clusters:
            raise ValueError("samples must be at least clusters")
        if self.dimension < 2:
            raise ValueError("dimension must be at least 2 for the report projection")
        if not math.isfinite(self.epochs) or self.epochs < 1:
            raise ValueError("epochs must be finite and at least 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


def make_data(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make independent training and held-out mixtures, plus common initial centers."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    centers = rng.normal(0, 4, (config.clusters, config.dimension))

    def sample(count: int) -> np.ndarray:
        labels = rng.integers(config.clusters, size=count)
        noise = rng.normal(0, 1, (count, config.dimension))
        return np.ascontiguousarray(centers[labels] + noise, dtype="float32")

    train = sample(config.samples)
    heldout = sample(max(100, config.samples // 4))
    initial = train[rng.choice(config.samples, config.clusters, replace=False)].copy()
    return train, heldout, initial


def exact_assignment(points: np.ndarray, centroids: np.ndarray) -> tuple[float, np.ndarray]:
    """Use the same exact Faiss L2 evaluation for both final models."""
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype="float32"))
    distances, labels = index.search(points, 1)
    return float(distances.mean(dtype="float64")), labels[:, 0]


def compare(config: Config) -> tuple[dict, dict]:
    """Train from identical starts and measure final loss independently of training."""
    config.validate()
    faiss.omp_set_num_threads(config.threads)
    train, heldout, initial = make_data(config)
    em = NumpyEmeans(config.dimension, config.clusters, nepochs=config.epochs, seed=config.seed)
    started = perf_counter()
    e_centroids, moving_labels = em.train(train, init_centroids=initial.copy())
    e_training_seconds = perf_counter() - started

    # Disabling Faiss's default sampling makes both algorithms see every row.
    lloyd = faiss.Kmeans(
        config.dimension, config.clusters, niter=config.lloyd_epochs, nredo=1,
        seed=config.seed, min_points_per_centroid=1,
        max_points_per_centroid=math.ceil(config.samples / config.clusters),
    )
    started = perf_counter()
    lloyd.train(train, init_centroids=initial.copy())
    lloyd_training_seconds = perf_counter() - started

    summary = {"config": asdict(config), "faiss_version": faiss.__version__, "models": {}}
    plots = {"points": train[:600, :2].tolist()}
    for name, centroids, seconds in (
        ("e-means", e_centroids, e_training_seconds),
        ("Lloyd", lloyd.centroids, lloyd_training_seconds),
    ):
        started = perf_counter()
        train_mse, labels = exact_assignment(train, centroids)
        heldout_mse, _ = exact_assignment(heldout, centroids)
        summary["models"][name] = {
            "training_seconds": seconds,
            "evaluation_seconds": perf_counter() - started,
            "train_mse": train_mse,
            "heldout_mse": heldout_mse,
            "cluster_counts": np.bincount(labels, minlength=config.clusters).tolist(),
            "centroids": centroids.tolist(),
        }
        plots[name] = {"centroids": centroids[:, :2].tolist(), "labels": labels[:600].tolist()}
    summary["emeans_recipe"] = em.recipe
    summary["emeans_revivals"] = em.n_splits
    summary["moving_labels_differ_from_final"] = int(np.count_nonzero(
        moving_labels != exact_assignment(train, e_centroids)[1]
    ))
    return summary, plots


def write_outputs(summary: dict, plots: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = directory / "index.html"
    report.write_text(render_report(summary, plots), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lloyd-epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--serve", action="store_true", help="keep a local report preview running")
    parser.add_argument("--port", type=int, default=8881)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    try:
        summary, plots = compare(config)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    report = write_outputs(summary, plots, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Report: {report.resolve()}", flush=True)
    if args.serve:
        handler = partial(SimpleHTTPRequestHandler, directory=str(args.output.resolve()))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Preview: http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
