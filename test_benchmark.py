"""Regression checks using real Faiss training, assignment and report output."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import faiss
import numpy as np

from benchmark import Config, compare, exact_assignment, make_data, write_outputs
from quickstart import run


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        faiss.omp_set_num_threads(1)
        self.config = Config(samples=160, dimension=4, clusters=4, epochs=2, lloyd_epochs=4)

    def test_exact_assignment_matches_independent_numpy_oracle(self):
        points = np.array([[0, 0], [3, 4], [-2, 1]], dtype="float32")
        centers = np.array([[1, 0], [4, 4]], dtype="float32")
        expected = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        mse, labels = exact_assignment(points, centers)
        np.testing.assert_array_equal(labels, expected.argmin(axis=1))
        self.assertAlmostEqual(mse, expected.min(axis=1).mean(dtype="float64"))

    def test_data_reproducible_and_independent(self):
        first = make_data(self.config)
        second = make_data(self.config)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0][:100], first[1]))

    def test_native_models_final_loss_and_counts(self):
        result, plots = compare(self.config)
        train, heldout, _ = make_data(self.config)
        for name, model in result["models"].items():
            centers = np.array(model["centroids"], dtype="float32")
            distances = ((train[:, None] - centers[None]) ** 2).sum(axis=2)
            self.assertAlmostEqual(model["train_mse"], distances.min(axis=1).mean(dtype="float64"), places=4)
            self.assertEqual(sum(model["cluster_counts"]), len(train))
            self.assertEqual(len(plots[name]["labels"]), len(train))
            heldout_distances = ((heldout[:, None] - centers[None]) ** 2).sum(axis=2)
            self.assertAlmostEqual(model["heldout_mse"], heldout_distances.min(axis=1).mean(dtype="float64"), places=4)
        self.assertEqual(result["emeans_recipe"]["backend"], "faiss")
        self.assertEqual(result["emeans_recipe"]["epoch_rows"], [160, 160])

    def test_training_reproducible(self):
        a, _ = compare(self.config)
        b, _ = compare(self.config)
        for name in a["models"]:
            np.testing.assert_array_equal(a["models"][name]["centroids"], b["models"][name]["centroids"])

    def test_fractional_passes_cover_final_metrics(self):
        result, _ = compare(replace(self.config, epochs=1.5))
        self.assertEqual(result["emeans_recipe"]["epoch_rows"], [80, 160])
        self.assertEqual(sum(result["models"]["e-means"]["cluster_counts"]), 160)

    def test_one_cluster(self):
        result, _ = compare(replace(self.config, clusters=1))
        self.assertEqual(result["models"]["Lloyd"]["cluster_counts"], [160])

    def test_report_is_standalone_and_json_roundtrips(self):
        summary, plots = compare(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = write_outputs(summary, plots, Path(directory))
            html = path.read_text(encoding="utf-8")
            self.assertIn("<svg", html)
            self.assertNotIn("<script", html)
            self.assertIn("Held-out loss", html)
            self.assertEqual(json.loads((path.parent / "results.json").read_text(encoding="utf-8")), summary)

    def test_invalid_config(self):
        for overrides in ({"samples": 0}, {"samples": 3}, {"dimension": 1}, {"clusters": 0},
                          {"epochs": float("nan")}, {"epochs": float("inf")}, {"epochs": .5},
                          {"seed": -1}, {"seed": 2**31}, {"threads": 0}, {"lloyd_epochs": 0},
                          {"samples": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(self.config, **overrides).validate()

    def test_quickstart_returns_real_arrays(self):
        centers, labels, distances, new_labels = run(200, 30, 4, 5, 7)
        self.assertEqual(centers.shape, (5, 4))
        self.assertEqual(labels.shape, (200,))
        self.assertEqual(distances.shape, (30,))
        self.assertEqual(new_labels.shape, (30,))
        self.assertTrue(np.isfinite(distances).all())
        self.assertTrue(((new_labels >= 0) & (new_labels < 5)).all())

    def test_quickstart_rejects_invalid_inputs(self):
        for args in ((0, 10, 4, 2), (10, 0, 4, 2), (10, 10, 0, 2), (10, 10, 4, 11)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                run(*args)


if __name__ == "__main__":
    unittest.main()
