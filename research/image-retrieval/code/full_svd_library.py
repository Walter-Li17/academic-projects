"""Reusable complete SVD records and normalized uncentered Gram descriptors.

The stored compact factors reconstruct the patch matrix (up to floating-point
roundoff). A Gram descriptor C = X.T @ X is NOT lossless: it discards U, including
patch order. Retrieval here compares these second-moment descriptors, not raw
patch matrices or an invented correspondence between patches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _factors(record):
    u, s, vh = (np.asarray(record[key]) for key in ("u", "s", "vh"))
    if u.ndim != 2 or s.ndim != 1 or vh.ndim != 2:
        raise ValueError("u, s, vh must be matrix, vector, matrix")
    if u.shape[1] != len(s) or vh.shape[0] != len(s):
        raise ValueError("factor dimensions are incompatible")
    if len(s) > min(u.shape[0], vh.shape[1]):
        raise ValueError("too many singular vectors for the matrix shape")
    if not len(s) or any(not np.isfinite(a).all() for a in (u, s, vh)):
        raise ValueError("factors must be nonempty and finite")
    if np.any(s < 0) or np.any(np.diff(s) > 0):
        raise ValueError("singular values must be nonnegative and descending")
    return u, s, vh


def reconstruct(record):
    """Reconstruct the original final-LayerNorm patch matrix, excluding CLS."""
    u, s, vh = _factors(record)
    return (u * s[None, :]) @ vh


def covariance_descriptor(record, rank=None):
    """Return normalized X.T@X, or its rank-k approximation, as a d×d matrix.

    No patch centering, no per-patch normalization, no U and no spatial matching.
    Ck = V_k diag(s_k**2) V_k.T; descriptor = Ck / ||Ck||_F.
    ``rank=None`` retains ALL compact SVD directions, including numerically tiny
    singular values. Scaling s by s[0] is harmless after normalization and avoids
    overflow. Numerical dtype is inherited from the stored factors.
    """
    _, s, vh = _factors(record)
    if rank is None:
        rank = len(s)
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= len(s):
        raise ValueError("rank must be an integer between one and the complete SVD length")
    if s[0] == 0:
        raise ValueError("the all-zero matrix has no normalized Gram descriptor")
    selected_s = s[:rank] / s[0]
    selected_vh = vh[:rank]
    descriptor = selected_vh.T @ ((selected_s ** 2)[:, None] * selected_vh)
    norm = np.linalg.norm(descriptor, "fro")
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid Gram norm")
    return descriptor / norm


def covariance_scores(query_descriptors, gallery_descriptors):
    """Normalized Frobenius inner products; output is [n_queries,n_gallery]."""
    q, g = np.asarray(query_descriptors), np.asarray(gallery_descriptors)
    if q.ndim != 3 or g.ndim != 3 or q.shape[1:] != g.shape[1:] or q.shape[1] != q.shape[2]:
        raise ValueError("inputs must have shapes [images,d,d] with the same d")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("descriptors must be finite")
    return q.reshape(len(q), -1) @ g.reshape(len(g), -1).T


def load_record(path, verify=True, expected_sha256=None):
    """Read one NPZ without pickle; optionally check its index-provided hash.

    ``verify`` checks complete dimensions, finite factors, sorted singular values,
    CLS shape, and metadata dimensions. File-integrity checking additionally uses
    ``expected_sha256``. FullSVDDatabase supplies that hash automatically.
    """
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as stored:
        required = {"u", "s", "vh", "cls", "metadata_json"}
        if not required.issubset(stored.files):
            raise ValueError(f"Incomplete SVD record: {path}")
        result = {key: stored[key] for key in ("u", "s", "vh", "cls")}
        result["metadata"] = json.loads(str(stored["metadata_json"].item()))
    if verify:
        u, s, vh = _factors(result)
        if len(s) != min(u.shape[0], vh.shape[1]):
            raise ValueError("stored record must contain the complete compact SVD")
        if result["cls"].shape != (vh.shape[1],) or not np.isfinite(result["cls"]).all():
            raise ValueError("CLS must be a finite feature vector")
        meta = result["metadata"]
        if meta.get("schema_version") != 1 or meta.get("matrix_shape") != [u.shape[0], vh.shape[1]]:
            raise ValueError("metadata schema or matrix shape mismatch")
    return result


class FullSVDDatabase:
    """Load complete factors by official query/gallery split index.

    Example::

        db = FullSVDDatabase('/project/data/roxford_dino_full_svd/index.json')
        query = db.load('query', 0)
        X = db.reconstruct('query', 0)
        C = db.descriptor('query', 0)         # all directions
        C32 = db.descriptor('query', 0, 32)   # rank 32
    """

    def __init__(self, index_path, verify_hashes=True):
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("schema_version") != 1:
            raise ValueError("unsupported database schema")
        self.verify_hashes = verify_hashes
        self.entries = {}
        for entry in self.index["entries"]:
            key = (entry["split"], int(entry["split_index"]))
            if key in self.entries:
                raise ValueError("duplicate split index in database")
            relative = Path(entry["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("record path must remain inside the database")
            self.entries[key] = entry

    def load(self, split, index):
        entry = self.entries[(split, index)]
        return load_record(self.root / entry["relative_path"], expected_sha256=entry["sha256"] if self.verify_hashes else None)

    def reconstruct(self, split, index):
        return reconstruct(self.load(split, index))

    def descriptor(self, split, index, rank=None):
        return covariance_descriptor(self.load(split, index), rank=rank)
