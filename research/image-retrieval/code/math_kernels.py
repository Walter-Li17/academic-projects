"""Bounded-memory distances for a finite DINO/SVD retrieval search.

All distances are lower-is-closer. ``negative_log_overlap`` is the stable
ranking key for Fubini--Study, not an eighth metric experiment. It can be +inf
for exactly orthogonal directions. Fubini--Study angles are for display only:
at high rank, rounding near pi/2 can erase real ordering differences.

Inputs q/g contain *columns* of orthonormal bases: [images, ambient_dim, kmax].
Singular-value powers in weighted_bures describe PSD eigenvalues, not factors.
"""

from __future__ import annotations

import numpy as np

ANGLE_METRICS = ("geodesic", "projection", "procrustes", "spectral", "fubini_study")
FS_RANKING_KEY = "negative_log_overlap"


def _array(x, name):
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError(f"{name} must contain real numbers")
    return a


def _validate_bases(q, g, k, *, check_orthonormal):
    q, g = _array(q, "q"), _array(g, "g")
    if q.ndim != 3 or g.ndim != 3:
        raise ValueError("q and g must have shape [images, ambient_dim, columns]")
    if q.shape[0] == 0 or g.shape[0] == 0:
        raise ValueError("q and g must contain at least one image")
    if q.shape[1] != g.shape[1]:
        raise ValueError("q and g must use the same ambient coordinates")
    if k < 1 or k > min(q.shape[2], g.shape[2], q.shape[1]):
        raise ValueError("requested k exceeds the available basis columns or ambient dimension")
    # Validation itself is chunked, to avoid a large temporary on memory maps.
    for name, a in (("q", q), ("g", g)):
        for start in range(0, len(a), 32):
            b = np.asarray(a[start:start + 32, :, :k], dtype=np.float64)
            if not np.isfinite(b).all():
                raise ValueError(f"{name} contains a nonfinite value")
            if check_orthonormal:
                gram = np.einsum("ndi,ndj->nij", b, b, optimize=True)
                if np.max(np.abs(gram - np.eye(k))) > 2e-5:
                    raise ValueError(f"{name} columns are not orthonormal")
    return q, g


def _cross_blocks(q, g, k, memory_mb):
    if not np.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("memory_mb must be finite and positive")
    # Conservative allowance for cross, SVD work and weighted-copy temporaries.
    pairs = max(1, int(memory_mb * 1024**2 // (8 * k * k * 4)))
    qstep = min(len(q), 4, pairs)
    gstep = min(len(g), max(1, pairs // qstep))
    for qi in range(0, len(q), qstep):
        qs = slice(qi, min(qi + qstep, len(q)))
        qb = np.asarray(q[qs, :, :k], dtype=np.float64)
        for gi in range(0, len(g), gstep):
            gs = slice(gi, min(gi + gstep, len(g)))
            gb = np.asarray(g[gs, :, :k], dtype=np.float64)
            h = np.einsum("qdi,gdj->qgij", qb, gb, optimize=True)
            yield qs, gs, h


def basis_distances(q, g, ks, *, memory_mb=32, check_orthonormal=True):
    """Return {(k, metric): [num_queries,num_gallery]} for each requested k.

    Computes the maximum-k cross product once per pair, then a distinct SVD for
    each leading k-by-k block. One SVD supplies all five angle metric families.
    Also returns (k, "negative_log_overlap") for stable Fubini--Study ranking.
    Runtime is bounded by the manifest, memory by blocks plus output arrays.
    """
    ks = list(ks)
    if not ks or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError("ks must be nonempty positive integers")
    ks = sorted(set(int(k) for k in ks))
    q, g = _validate_bases(q, g, max(ks), check_orthonormal=check_orthonormal)
    result = {
        (k, metric): np.empty((len(q), len(g)), dtype=np.float64)
        for k in ks for metric in (*ANGLE_METRICS, FS_RANKING_KEY)
    }
    for qs, gs, hmax in _cross_blocks(q, g, max(ks), memory_mb):
        for k in ks:
            s = np.linalg.svd(hmax[..., :k, :k], compute_uv=False)
            if np.max(s) > 1 + 3e-5:
                raise ValueError("cross-Gram singular values materially exceed one")
            s = np.clip(s, 0, 1)
            theta = np.arccos(s)
            with np.errstate(divide="ignore"):
                neg_log_overlap = -np.log(s).sum(axis=-1)
            vals = {
                "geodesic": np.linalg.norm(theta, axis=-1),
                "projection": np.sqrt(np.maximum(0, k - (s * s).sum(axis=-1))),
                "procrustes": np.sqrt(np.maximum(0, 2 * k - 2 * s.sum(axis=-1))),
                "spectral": np.sqrt(np.maximum(0, 1 - s[..., -1] ** 2)),
                "fubini_study": np.arccos(np.exp(-neg_log_overlap)),
                FS_RANKING_KEY: neg_log_overlap,
            }
            for metric, value in vals.items():
                result[k, metric][qs, gs] = value
    return result


def _normalized_weights(s, n, k, power, name):
    s = _array(s, name)
    if s.ndim != 2 or s.shape[0] != n or s.shape[1] < k:
        raise ValueError(f"{name} must have shape [images, at_least_k_singular_values]")
    s = np.asarray(s[:, :k], dtype=np.float64)
    if not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if np.any(s.max(axis=1) == 0):
        raise ValueError(f"{name} contains a zero-energy descriptor")
    # Scaling before exponentiation avoids overflow for large singular values.
    scaled = s / s.max(axis=1, keepdims=True)
    w = scaled ** power
    return w / w.sum(axis=1, keepdims=True)


def weighted_bures(q, g, sq, sg, k, power, *, memory_mb=32, check_orthonormal=True):
    """Trace-normalized PSD Bures distance with eigenvalues proportional σ^power.

    q/g are V bases, sq/sg their paired original-matrix singular values.
    Different patch counts are allowed since V shares feature coordinates.
    power=2 is a truncated second moment (covariance if features centered).
    power=1 is a spectral square-root / power-normalized PSD representation.
    Finite positive powers are supported, though the manifest should be fixed.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    q, g = _validate_bases(q, g, int(k), check_orthonormal=check_orthonormal)
    wq = _normalized_weights(sq, len(q), k, power, "sq")
    wg = _normalized_weights(sg, len(g), k, power, "sg")
    result = np.empty((len(q), len(g)), dtype=np.float64)
    for qs, gs, h in _cross_blocks(q, g, k, memory_mb):
        weighted = np.sqrt(wq[qs])[:, None, :, None] * h * np.sqrt(wg[gs])[None, :, None, :]
        fidelity = np.linalg.svd(weighted, compute_uv=False).sum(axis=-1)
        if np.max(fidelity) > 1 + 3e-5:
            raise ValueError("normalized Bures fidelity materially exceeds one")
        result[qs, gs] = np.sqrt(np.maximum(0, 2 - 2 * np.clip(fidelity, 0, 1)))
    return result


def rmse(q, g, *, memory_mb=32):
    """Pairwise entrywise RMSE; inputs [images, ...identical matrix shape...].

    No implicit per-image feature normalization is performed. The normalization
    is division by sqrt(number of matrix entries), matching ordinary RMSE.
    """
    q, g = _array(q, "q"), _array(g, "g")
    if q.ndim < 2 or g.ndim != q.ndim or q.shape[1:] != g.shape[1:]:
        raise ValueError("RMSE requires matching nonempty matrix shapes")
    if len(q) == 0 or len(g) == 0 or np.prod(q.shape[1:]) == 0:
        raise ValueError("RMSE inputs must be nonempty")
    if not np.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("memory_mb must be finite and positive")
    q = q.reshape(len(q), -1)
    g = g.reshape(len(g), -1)
    n = q.shape[1]
    step = max(1, int(memory_mb * 1024**2 // (8 * n * 3)))
    out = np.empty((len(q), len(g)), dtype=np.float64)
    # Direct differences avoid catastrophic cancellation for almost equal M.
    for qi in range(len(q)):
        qb = np.asarray(q[qi], dtype=np.float64)
        if not np.isfinite(qb).all():
            raise ValueError("q contains a nonfinite value")
        for gi in range(0, len(g), step):
            gb = np.asarray(g[gi:gi + step], dtype=np.float64)
            if not np.isfinite(gb).all():
                raise ValueError("g contains a nonfinite value")
            diff = gb - qb
            out[qi, gi:gi + step] = np.sqrt(np.einsum("ij,ij->i", diff, diff) / n)
    return out
