"""Small NumPy reference for spectral image descriptors and ROxford evaluation.

No model inference or file I/O occurs in this module. Ground-truth image IDs must
index the complete gallery in the official ``imlist`` order, starting at zero.
Evaluation follows the author's Revisited Oxford/Paris ``evaluate.py`` and
``example_evaluate.py`` (CVPR 2018), including trapezoidal AP and junk removal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


def spectral_factor(singular_values, vh, alpha=0.0, k=None):
    """Return W such that C = W.T @ W has unit Frobenius norm.

    ``vh`` has right singular vectors as ROWS (NumPy SVD's third return value).
    For retained positive singular values s, w = s**alpha / ||s**alpha||_2
    and W = diag(sqrt(w)) @ vh. alpha=0 gives a normalized subspace projector;
    alpha=1 uses singular-value weights; alpha=2 uses covariance-like weights.
    Input rows must be orthonormal. A retained zero singular value is rejected:
    its right singular vector is arbitrary and is not evidence about the image.

    This representation is invariant to sign changes and rotations within a
    retained equal-weight eigenspace. Cutting THROUGH a tied singular-value
    group can destroy invariance, as with any fixed-rank truncation.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    v = np.asarray(vh, dtype=np.float64)
    if s.ndim != 1 or v.ndim != 2 or len(s) != v.shape[0]:
        raise ValueError("singular_values must have one entry per row of vh")
    if k is None:
        k = len(s)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    if not 1 <= k <= len(s):
        raise ValueError("k must be between one and the supplied number of vectors")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    s, v = s[:k], v[:k]
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(v)):
        raise ValueError("singular values and singular vectors must be finite")
    if np.any(s <= 0):
        raise ValueError("retained singular values must be strictly positive")
    if np.any(np.diff(s) > 0):
        raise ValueError("singular values must be in descending order")
    if not np.allclose(v @ v.T, np.eye(k), atol=2e-5, rtol=2e-5):
        raise ValueError("rows of vh must be orthonormal")
    # Rescaling before the power avoids overflow and cancels in normalization.
    weights = (s / s[0]) ** alpha
    weights /= np.linalg.norm(weights)
    return np.sqrt(weights)[:, None] * v


def spectral_similarity(query_factor, gallery_factor):
    """Compute <Cq,Cg>_F = ||Wq @ Wg.T||_F**2 using small factors.

    For factors returned by spectral_factor, this is a positive-semidefinite
    kernel between zero and one, with self-similarity one. Different retained
    ranks are allowed, provided both images use the same feature coordinates.
    """
    q = np.asarray(query_factor, dtype=np.float64)
    g = np.asarray(gallery_factor, dtype=np.float64)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("factors must be two-dimensional with equal feature dimension")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(g)):
        raise ValueError("factors must contain finite values")
    cross = q @ g.T
    return float(np.sum(cross * cross))


def pairwise_spectral_scores(query_factors, gallery_factors, batch_size=64):
    """Score fixed-rank factors, returning a [n_queries, n_gallery] array.

    Inputs are arrays [n_images, rank, feature_dimension]. Query and gallery
    ranks can differ. Work is batched over gallery images to limit memory.
    This is a transparent CPU reference, not an optimized large-scale index.
    """
    q = np.asarray(query_factors, dtype=np.float64)
    g = np.asarray(gallery_factors, dtype=np.float64)
    if q.ndim != 3 or g.ndim != 3 or q.shape[2] != g.shape[2]:
        raise ValueError("factors must have shape [images, rank, shared_feature_dimension]")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(g)):
        raise ValueError("factors must contain finite values")
    scores = np.empty((len(q), len(g)), dtype=np.float64)
    for qi, factor in enumerate(q):
        for start in range(0, len(g), batch_size):
            cross = factor[None, :, :] @ g[start:start + batch_size].transpose(0, 2, 1)
            scores[qi, start:start + batch_size] = np.sum(cross * cross, axis=(1, 2))
    return scores


def average_precision(positive_ranks, n_positives):
    """Official trapezoidal AP from zero-based positive ranks AFTER junk removal.

    This is not sklearn's average_precision_score. For example, one relevant
    image at zero-based rank 1 has AP 0.25, rather than reciprocal rank 0.5.
    Empty positive ground truth has undefined AP (NaN).
    """
    if isinstance(n_positives, (bool, np.bool_)) or not isinstance(n_positives, (int, np.integer)) or n_positives < 0:
        raise ValueError("n_positives must be a nonnegative integer")
    ranks = np.asarray(positive_ranks)
    if ranks.ndim != 1 or (ranks.size and ranks.dtype.kind not in "iu"):
        raise ValueError("positive_ranks must be a one-dimensional integer array")
    ranks = ranks.astype(np.int64)
    if len(ranks) > n_positives or np.any(ranks < 0) or np.any(np.diff(ranks) <= 0):
        raise ValueError("positive ranks must be distinct, sorted, nonnegative, and no more than n_positives")
    if n_positives == 0:
        return float("nan")
    ap = 0.0
    for j, rank in enumerate(ranks):
        precision_before = 1.0 if rank == 0 else j / float(rank)
        precision_after = (j + 1) / float(rank + 1)
        ap += (precision_before + precision_after) / (2.0 * n_positives)
    return float(ap)


def _index_array(record, key, gallery_size):
    array = np.asarray(record.get(key, []))
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"ground-truth {key} must be a one-dimensional integer array")
    array = array.astype(np.int64)
    if np.any(array < 0) or np.any(array >= gallery_size):
        raise ValueError(f"ground-truth {key} index outside the complete gallery")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"duplicate index in ground-truth {key}")
    return array


def evaluate_protocol(scores, gnd, protocol="medium"):
    """Evaluate complete-gallery similarity scores under Easy/Medium/Hard.

    Higher scores are better. ``scores`` is [n_queries, n_gallery], whereas
    the author's evaluator expects rankings transposed to [n_gallery,n_queries].
    Ties are broken stably in gallery order. ``gnd`` accepts either the list of
    per-query annotations or a readable-JSON dict containing that list at gnd.

    Returns a dict with mAP (fraction, not percent), per-query aps (NaN for
    queries without positives), rankings [n_queries,n_gallery], n_evaluated,
    n_queries, n_gallery, and protocol. If every query has no positives, mAP
    is NaN rather than dividing by zero. Negatives remain in the ranking;
    junk is excluded separately for each query and each protocol.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite matrix [n_queries, n_gallery]")
    if isinstance(gnd, Mapping):
        gnd = gnd["gnd"]
    if len(gnd) != scores.shape[0]:
        raise ValueError("one ground-truth record is required per query")
    aliases = {"e": "easy", "easy": "easy", "m": "medium", "medium": "medium", "h": "hard", "hard": "hard"}
    protocol = aliases.get(str(protocol).lower())
    if protocol is None:
        raise ValueError("protocol must be Easy, Medium, or Hard")
    rankings = np.argsort(-scores, axis=1, kind="stable")
    aps = np.full(scores.shape[0], np.nan, dtype=np.float64)
    for qi, record in enumerate(gnd):
        easy, hard, junk = (_index_array(record, key, scores.shape[1]) for key in ("easy", "hard", "junk"))
        if len(np.unique(np.concatenate((easy, hard, junk)))) != len(easy) + len(hard) + len(junk):
            raise ValueError("easy, hard, and junk ground-truth sets must be disjoint")
        if protocol == "easy":
            positives, excluded = easy, np.concatenate((junk, hard))
        elif protocol == "medium":
            positives, excluded = np.concatenate((easy, hard)), junk
        else:
            positives, excluded = hard, np.concatenate((junk, easy))
        if len(positives) == 0:
            continue
        cleaned_ranking = rankings[qi][~np.isin(rankings[qi], excluded)]
        positive_ranks = np.flatnonzero(np.isin(cleaned_ranking, positives))
        aps[qi] = average_precision(positive_ranks, len(positives))
    valid = np.isfinite(aps)
    return {
        "mAP": float(np.mean(aps[valid])) if np.any(valid) else float("nan"),
        "aps": aps,
        "rankings": rankings,
        "n_evaluated": int(np.sum(valid)),
        "n_queries": scores.shape[0],
        "n_gallery": scores.shape[1],
        "protocol": protocol,
    }
