"""Math invariants and comparison against the unmodified author evaluator.

Run: python -m unittest discover -s PATH_TO_THIS_DIRECTORY -v
Optional paths: ROXFORD_REFERENCE_EVALUATOR and ROXFORD_READABLE_GND.
Tests never load model weights, execute inference, or write into the dataset.
"""

import json
import os
from pathlib import Path
import unittest
import numpy as np

from retrieval_math import (
    average_precision, evaluate_protocol, pairwise_spectral_scores,
    spectral_factor, spectral_similarity,
)


DATASET = Path(__file__).resolve().parents[1] / "data" / "roxford5k"
REFERENCE = Path(os.environ.get("ROXFORD_REFERENCE_EVALUATOR", str(DATASET / "reference_code/revisitop-master/python/evaluate.py")))
READABLE_GND = Path(os.environ.get("ROXFORD_READABLE_GND", str(DATASET / "gnd_roxford5k_readable.json")))


def load_official_compute_map():
    if not REFERENCE.is_file():
        raise unittest.SkipTest(f"Author evaluator unavailable: {REFERENCE}")
    namespace = {"__name__": "official_roxford_evaluator"}
    # Compile/read-only rather than importlib: do not create a dataset pycache.
    exec(compile(REFERENCE.read_text(), str(REFERENCE), "exec"), namespace)
    if not hasattr(np, "in1d"):
        # New NumPy releases remove the author's deprecated spelling. This
        # compatibility proxy preserves membership behavior without editing
        # either the reference file or the global NumPy module.
        class NumpyCompatibility:
            @staticmethod
            def in1d(a, b):
                return np.isin(a, b)

            def __getattr__(self, name):
                return getattr(np, name)
        namespace["np"] = NumpyCompatibility()
    return namespace["compute_map"]


def official_annotations(gnd, protocol):
    result = []
    for entry in gnd:
        easy, hard, junk = (np.asarray(entry[key], dtype=np.int64) for key in ("easy", "hard", "junk"))
        if protocol == "easy":
            ok, excluded = easy, np.concatenate((junk, hard))
        elif protocol == "medium":
            ok, excluded = np.concatenate((easy, hard)), junk
        else:
            ok, excluded = hard, np.concatenate((junk, easy))
        result.append({"ok": ok, "junk": excluded})
    return result


class SpectralMathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(38271)

    def basis(self, rank=3, dimension=8):
        return np.linalg.qr(self.rng.normal(size=(dimension, rank)))[0].T

    def test_dense_and_factor_formulas_agree(self):
        for alpha in (0, 1, 2):
            q = spectral_factor([5, 2, 1], self.basis(), alpha)
            g = spectral_factor([7, 3], self.basis(rank=2), alpha)
            cq, cg = q.T @ q, g.T @ g
            self.assertAlmostEqual(spectral_similarity(q, g), float(np.sum(cq * cg)), places=13)
            self.assertAlmostEqual(np.linalg.norm(cq, "fro"), 1.0, places=13)

    def test_sign_invariance(self):
        v = self.basis()
        other = self.basis()
        for alpha in (0, 1, 2):
            original = spectral_factor([8, 3, 1], v, alpha)
            flipped = spectral_factor([8, 3, 1], v * np.array([-1, 1, -1])[:, None], alpha)
            gallery = spectral_factor([9, 3, 2], other, alpha)
            np.testing.assert_allclose(original.T @ original, flipped.T @ flipped, atol=1e-14)
            self.assertAlmostEqual(spectral_similarity(original, gallery), spectral_similarity(flipped, gallery), places=13)

    def test_degenerate_eigenspace_rotation_invariance(self):
        v = self.basis()
        angle = 0.417
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        for alpha in (0, 1, 2):
            a = spectral_factor([8, 2, 2], v, alpha)
            b = spectral_factor([8, 2, 2], rotation @ v, alpha)
            np.testing.assert_allclose(a.T @ a, b.T @ b, atol=1e-14)

    def test_alpha_zero_ignores_all_basis_rotations(self):
        v = self.basis()
        rotation = np.linalg.qr(self.rng.normal(size=(3, 3)))[0]
        a = spectral_factor([8, 3, 1], v, 0)
        b = spectral_factor([8, 3, 1], rotation @ v, 0)
        np.testing.assert_allclose(a.T @ a, b.T @ b, atol=1e-14)

    def test_kernel_psd_normalized_diagonal_and_bounds(self):
        for alpha in (0, 1, 2):
            factors = np.stack([spectral_factor([7, 3, 1], self.basis(), alpha) for _ in range(7)])
            scores = pairwise_spectral_scores(factors, factors, batch_size=3)
            np.testing.assert_allclose(scores, scores.T, atol=1e-14)
            np.testing.assert_allclose(np.diag(scores), 1.0, atol=1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(scores).min(), -1e-12)
            self.assertGreaterEqual(scores.min(), -1e-14)
            self.assertLessEqual(scores.max(), 1 + 1e-12)
            self.assertAlmostEqual(scores[1, 5], spectral_similarity(factors[1], factors[5]), places=13)

    def test_alpha_zero_recovers_normalized_projector_similarity(self):
        q, g = self.basis(), self.basis()
        actual = spectral_similarity(spectral_factor([9, 3, 1], q), spectral_factor([6, 2, 1], g))
        expected = np.linalg.norm(q @ g.T, "fro") ** 2 / 3
        self.assertAlmostEqual(actual, expected, places=13)

    def test_scale_invariance_and_truncation(self):
        v = self.basis()
        for alpha in (0, 1, 2):
            a = spectral_factor([8, 3, 1], v, alpha, k=2)
            b = spectral_factor(np.array([8, 3, 1]) * 1e150, v, alpha, k=2)
            np.testing.assert_allclose(a, b, atol=1e-14)
            self.assertEqual(a.shape, (2, 8))

    def test_rejects_arbitrary_nullspace_and_nonorthogonal_rows(self):
        with self.assertRaises(ValueError):
            spectral_factor([1, 0], np.eye(2), alpha=0)
        with self.assertRaises(ValueError):
            spectral_factor([2, 1], np.ones((2, 3)))


class EvaluationTests(unittest.TestCase):
    def test_trapezoidal_ap_not_precision_at_positive(self):
        self.assertEqual(average_precision([0], 1), 1.0)
        self.assertEqual(average_precision([1], 1), 0.25)
        self.assertAlmostEqual(average_precision([0, 2], 2), 19 / 24)

    def test_junk_removal_and_stable_ties(self):
        # All equal scores retain gallery IDs 0,1,2,3,4. Easy excludes hard ID
        # 1 and junk ID 0, leaving negative 2 before relevant 3 (AP = .25).
        scores = np.ones((1, 5))
        gnd = [{"easy": [3], "hard": [1], "junk": [0]}]
        e = evaluate_protocol(scores, gnd, "Easy")
        np.testing.assert_array_equal(e["rankings"], [[0, 1, 2, 3, 4]])
        self.assertAlmostEqual(e["mAP"], 0.25)
        self.assertAlmostEqual(evaluate_protocol(scores, gnd, "Medium")["mAP"], 19 / 24)
        self.assertAlmostEqual(evaluate_protocol(scores, gnd, "Hard")["mAP"], 1.0)

    def test_empty_positive_queries_are_excluded(self):
        scores = np.array([[3, 2, 1], [3, 2, 1]])
        gnd = [{"easy": [], "hard": [], "junk": []}, {"easy": [0], "hard": [], "junk": []}]
        result = evaluate_protocol(scores, {"gnd": gnd}, "medium")
        self.assertTrue(np.isnan(result["aps"][0]))
        self.assertEqual(result["mAP"], 1.0)
        self.assertEqual(result["n_evaluated"], 1)
        empty = evaluate_protocol(scores[:1], gnd[:1], "hard")
        self.assertTrue(np.isnan(empty["mAP"]))
        self.assertEqual(empty["n_evaluated"], 0)

    def test_official_evaluator_agreement_with_junk_and_empty_queries(self):
        official = load_official_compute_map()
        scores = np.array([[9, 1, 8, 3, 7, 2, 6, 5], [2, 8, 3, 7, 4, 6, 1, 5], [1, 3, 5, 7, 8, 6, 4, 2]])
        gnd = [
            {"easy": [3, 6], "hard": [1], "junk": [0, 4]},
            {"easy": [], "hard": [6, 7], "junk": [1, 3]},
            {"easy": [1, 2], "hard": [], "junk": [0, 4, 6]},
        ]
        for protocol in ("easy", "medium", "hard"):
            result = evaluate_protocol(scores, gnd, protocol)
            expected_map, expected_aps, _, _ = official(result["rankings"].T, official_annotations(gnd, protocol))
            self.assertAlmostEqual(result["mAP"], expected_map, places=14)
            np.testing.assert_allclose(result["aps"], expected_aps, atol=1e-14, equal_nan=True)

    def test_real_annotations_match_official_for_random_full_rankings(self):
        official = load_official_compute_map()
        if not READABLE_GND.is_file():
            self.skipTest(f"Readable annotations unavailable: {READABLE_GND}")
        payload = json.loads(READABLE_GND.read_text())
        gnd = payload["gnd"]
        # Read imlist when present; official ROxford has 4,993 gallery images.
        ng = len(payload["imlist"]) if "imlist" in payload else 4993
        scores = np.random.default_rng(7304).standard_normal((len(gnd), ng))
        for protocol in ("easy", "medium", "hard"):
            result = evaluate_protocol(scores, gnd, protocol)
            expected_map, expected_aps, _, _ = official(result["rankings"].T, official_annotations(gnd, protocol))
            self.assertAlmostEqual(result["mAP"], expected_map, places=14)
            np.testing.assert_allclose(result["aps"], expected_aps, atol=1e-14, equal_nan=True)

    def test_invalid_ground_truth_or_scores_fail_loudly(self):
        with self.assertRaises(ValueError):
            evaluate_protocol([[1, 2]], [{"easy": [2]}])
        with self.assertRaises(ValueError):
            evaluate_protocol([[1, 2]], [{"easy": [1], "hard": [1]}])
        with self.assertRaises(ValueError):
            evaluate_protocol([[1, float("nan")]], [{"easy": [1]}])
        with self.assertRaises(ValueError):
            evaluate_protocol([[1, 2]], [{"easy": [1.0]}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
