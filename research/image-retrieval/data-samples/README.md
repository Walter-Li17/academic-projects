# Derived feature samples

These two records come from DINO features extracted for queries in [Revisited Oxford](https://cmp.felk.cvut.cz/revisitop/). They contain numerical feature data, not the original photographs or model weights.

- [all_souls_000013](query_all_souls_000013.npz): a common-grid `matrix` of shape `(196, 384)`, `u32`, `v32`, singular values `s`, a `cls` vector, and `metadata_json`.
- [oxford_002904](query_oxford_002904_full_svd.npz): compact full SVD factors `u`, `s`, and `vh`, a `cls` vector, and `metadata_json`. The feature matrix has shape `(196, 384)` and can be reconstructed as `(u * s) @ vh`.

The metadata records image identifiers, feature dimensions, and numerical provenance. The second record's saved relative reconstruction error is approximately `1.27e-6`.

Inspect a record with NumPy:

```python
import numpy as np

with np.load("research/image-retrieval/data-samples/query_oxford_002904_full_svd.npz", allow_pickle=False) as record:
    matrix = (record["u"] * record["s"]) @ record["vh"]
    print(matrix.shape)
    print(record["metadata_json"].item())
```

For the source images, dataset terms, and original evaluation protocol, consult the benchmark's official page. The complete feature cache and image corpus are not included here.
