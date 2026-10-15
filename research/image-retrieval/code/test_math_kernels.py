"""Analytical and independent-reference tests; no benchmark experiments."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from math_kernels import ANGLE_METRICS, FS_RANKING_KEY, basis_distances, weighted_bures, rmse


def random_basis(rng, d, k):
    return np.linalg.qr(rng.normal(size=(d, k)), mode="reduced")[0]


def sqrt_psd(a):
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.maximum(0, w))) @ v.T


class MetricTests(unittest.TestCase):
    def test_known_principal_angles(self):
        p = np.eye(4)[:, :2]
        a, b = 0.2, 0.8
        q = np.array([[np.cos(a), 0], [0, np.cos(b)], [np.sin(a), 0], [0, np.sin(b)]])
        r = basis_distances(p[None], q[None], [1, 2])
        expected = {
            "geodesic": np.hypot(a, b),
            "projection": np.hypot(np.sin(a), np.sin(b)),
            "procrustes": 2 * np.hypot(np.sin(a / 2), np.sin(b / 2)),
            "spectral": np.sin(b),
            "fubini_study": np.arccos(np.cos(a) * np.cos(b)),
            FS_RANKING_KEY: -np.log(np.cos(a) * np.cos(b)),
        }
        for metric, value in expected.items():
            assert_allclose(r[2, metric], [[value]], atol=1e-12)
        assert_allclose(r[1, "geodesic"], [[a]], atol=1e-12)

    def test_basis_rotation_and_sign_invariance(self):
        rng = np.random.default_rng(83)
        p, q = random_basis(rng, 9, 3), random_basis(rng, 9, 3)
        r1, r2 = random_basis(rng, 3, 3), random_basis(rng, 3, 3)
        original = basis_distances(p[None], q[None], [3])
        changed = basis_distances((p @ r1)[None], (q @ r2)[None], [3])
        for key in original:
            assert_allclose(original[key], changed[key], atol=1e-12)

    def test_identical_and_orthogonal_subspaces(self):
        p = np.eye(6)[:, :3]
        q = np.eye(6)[:, 3:]
        same = basis_distances(p[None], p[None], [3])
        ortho = basis_distances(p[None], q[None], [3])
        for metric in ANGLE_METRICS:
            assert_allclose(same[3, metric], 0, atol=1e-12)
        self.assertTrue(np.isinf(ortho[3, FS_RANKING_KEY][0, 0]))
        assert_allclose(ortho[3, "geodesic"], np.sqrt(3) * np.pi / 2)
        assert_allclose(ortho[3, "projection"], np.sqrt(3))

    def test_projection_independent_projector_formula(self):
        rng = np.random.default_rng(2)
        p, q = random_basis(rng, 7, 3), random_basis(rng, 7, 3)
        expected = np.linalg.norm(p @ p.T - q @ q.T, "fro") / np.sqrt(2)
        actual = basis_distances(p[None], q[None], [3])[3, "projection"][0, 0]
        assert_allclose(actual, expected, atol=1e-12)

    def test_rank_one_equivalent_rankings(self):
        p = np.array([1.0, 0.0])[None, :, None]
        angles = np.array([0.9, 0.1, 1.2, 0.5])
        g = np.stack([np.cos(angles), np.sin(angles)], axis=1)[..., None]
        r = basis_distances(p, g, [1])
        for key, d in r.items():
            assert_allclose(np.argsort(d[0]), np.argsort(angles))
        for power in (1, 2):
            d = weighted_bures(p, g, [[10]], [[2], [3], [8], [1]], 1, power)
            assert_allclose(np.argsort(d[0]), np.argsort(angles))

    def test_fubini_ranking_key_prevents_saturation_ties(self):
        k = 32
        p = np.eye(2 * k)[:, :k]
        g = np.stack([np.vstack([c * np.eye(k), np.sqrt(1 - c*c) * np.eye(k)]) for c in (0.1, 0.2)])
        r = basis_distances(p[None], g, [k])
        # Both rounded display angles are pi/2, but ordering is still available.
        self.assertEqual(r[k, "fubini_study"][0, 0], r[k, "fubini_study"][0, 1])
        self.assertGreater(r[k, FS_RANKING_KEY][0, 0], r[k, FS_RANKING_KEY][0, 1])

    def test_chunked_outputs_and_rank_prefixes(self):
        rng = np.random.default_rng(109)
        p = np.stack([random_basis(rng, 8, 4) for _ in range(3)])
        g = np.stack([random_basis(rng, 8, 4) for _ in range(5)])
        a = basis_distances(p, g, [1, 2, 4], memory_mb=0.001)
        b = basis_distances(p, g, [1, 2, 4], memory_mb=32)
        c = basis_distances(p[:, :, :2], g[:, :, :2], [2])
        for key in a:
            assert_allclose(a[key], b[key], atol=1e-12)
        for key in c:
            assert_allclose(a[key], c[key], atol=1e-12)

    def test_bures_matches_dense_psd_formula(self):
        rng = np.random.default_rng(192)
        p, q = random_basis(rng, 6, 3), random_basis(rng, 6, 3)
        sp, sq = np.array([9., 3., 1.]), np.array([5., 4., 2.])
        for power in (1, 2):
            wa, wb = sp**power, sq**power
            wa, wb = wa / wa.sum(), wb / wb.sum()
            a, b = (p * wa) @ p.T, (q * wb) @ q.T
            sa = sqrt_psd(a)
            dense = np.sqrt(max(0, np.trace(a) + np.trace(b) - 2 * np.trace(sqrt_psd(sa @ b @ sa))))
            actual = weighted_bures(p[None], q[None], sp[None], sq[None], 3, power)[0, 0]
            assert_allclose(actual, dense, atol=5e-8)

    def test_bures_scale_sign_invariance_and_equal_weight_limit(self):
        rng = np.random.default_rng(142)
        p, q = random_basis(rng, 8, 3), random_basis(rng, 8, 3)
        s = np.array([[4., 2., 1.]])
        a = weighted_bures(p[None], q[None], s, s, 3, 2)
        b = weighted_bures((-p)[None], q[None], s * 1e100, s * 1e-100, 3, 2)
        assert_allclose(a, b, atol=1e-12)
        uniform = weighted_bures(p[None], q[None], np.ones((1, 3)), np.ones((1, 3)), 3, 2)
        procrustes = basis_distances(p[None], q[None], [3])[3, "procrustes"]
        assert_allclose(uniform, procrustes / np.sqrt(3), atol=1e-12)

    def test_rmse_numeric_example_and_small_difference(self):
        a = np.array([[[1., 2.], [3., 4.]]])
        b = np.array([[[1., 4.], [3., 6.]], [[1., 2.], [3., 4.]]])
        assert_allclose(rmse(a, b, memory_mb=0.001), [[np.sqrt(2), 0]])
        a = np.full((1, 5), 1e8)
        assert_allclose(rmse(a, a + 1), [[1.]])

    def test_invalid_input_guards(self):
        p = np.eye(3)[None, :, :2]
        with self.assertRaises(ValueError):
            basis_distances(2 * p, p, [2])
        with self.assertRaises(ValueError):
            basis_distances(p, np.eye(4)[None, :, :2], [2])
        with self.assertRaises(ValueError):
            basis_distances(p, p, [3])
        with self.assertRaises(ValueError):
            basis_distances(p * np.nan, p, [2])
        with self.assertRaises(ValueError):
            weighted_bures(p, p, [[0, 0]], [[1, 1]], 2, 2)
        with self.assertRaises(ValueError):
            weighted_bures(p, p, [[-1, 1]], [[1, 1]], 2, 2)
        with self.assertRaises(ValueError):
            rmse(np.ones((1, 2, 3)), np.ones((1, 3, 2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
