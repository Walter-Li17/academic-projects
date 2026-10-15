# Experimental results

| File | Scope | Outcome |
|---|---|---|
| [Subspace search](subspace_search_696.csv) | 696 planned entries: 600 scored, 84 equivalent, 12 invalid | No configuration improved both Oxford Medium and Hard over matched CLS. |
| [Patch concatenation](patch_concatenation_15.csv) | 15 predefined variants | No dual-metric Oxford improvement. |
| [PCA and whitening](representation_10_oxford.csv) | 10 targeted variants | Four exploratory Oxford dual-metric improvements. |
| [Per-query changes](representation_query_changes.csv) | Query-level AP comparisons | Additional detail for inspecting the representation variants. |

The strongest Oxford candidate in the final batch was **03** (Medium 34.4767%, Hard 8.7316%); it was not tested on Paris. The fixed Paris candidate was **01**, a different method:

| Dataset | Method | Medium mAP | Hard mAP |
|---|---|---:|---:|
| Oxford | CLS reference | 32.2505% | 7.8191% |
| Oxford | Centered CLS, candidate 01 | 32.4367% | 8.2089% |
| Paris | CLS reference | 61.8137% | 34.6514% |
| Paris | Fixed candidate 01 | 61.0649% | 34.8190% |

Paris did not confirm improvement on both metrics. Oxford was used repeatedly for exploration; its selected scores are not independent confirmation. A finite search of these choices also does not rule out all possible SVD or local-feature methods.

The CSV files preserve the recorded experiment outcomes. Timing values are incremental measurements with reused computations and should not be treated as fair standalone speed benchmarks.
