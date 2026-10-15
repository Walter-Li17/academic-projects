# Image retrieval with DINO features and SVD

Exploratory master's thesis work supervised by Vahan Huroyan, Saint Louis University. Research materials below document current progress; they are not a published paper.

## Read and explore

- **[Research presentation](Image_Retrieval_Research.pdf)** — 27 pages explaining the feature representations, comparison design, and next research direction.
- **[Candidate 03 study](Candidate_03_Study.pdf)** — Six-page supplement on gallery-fitted PCA and soft whitening.
- **[Results](results/)** — Completed comparison tables and interpretation.
- **[Code](code/)** — NumPy implementations of the numerical comparison and evaluation functions, with tests.
- **[Data samples](data-samples/)** — Two real derived feature records, including matrix and SVD factors.
- **[Reading references](REFERENCES.md)** — DINO, Grassmann pooling, evaluation, and mathematical background.
- **[Interactive demo](https://dino-oxford-retrieval.liruoshui0710.chatgpt.site/)** — View image-retrieval rankings.

## What the evidence supports

The 696-entry subspace comparison contains 600 scored independent configurations, 84 equivalent entries, and 12 invalid configurations. None beat the matched CLS baseline on both Oxford Medium and Hard mAP. A separate 15-variant patch-concatenation study also found no such improvement.

A ten-variant PCA/whitening study found four exploratory Oxford improvements. Candidate 03 reached Medium 34.4767% and Hard 8.7316%, compared with CLS at 32.2505% and 7.8191%. Candidate 03 was **not evaluated on Paris**. Candidate 01 was fixed and tested on Paris; it did not improve both metrics. These results do not establish a generalizable improvement.

The planned Grassmann-pooling replication will add learned dimensionality reduction, SVD pooling, and a projection-similarity classifier. That next phase is not yet completed.

The implementation and numerical checking were developed with AI assistance. The existing DINO, SVD, and geometric methods are credited in the references; this repository does not claim their invention.

## Run the numerical checks

With Python and NumPy installed, run from the repository root:

```sh
python -m unittest discover -s research/image-retrieval/code -p 'test_*.py' -v
```

The standalone tests do not download model weights or run DINO inference. Two optional tests compare with the original benchmark evaluator and require external evaluation files. Without those files they are reported as skipped. Set `ROXFORD_REFERENCE_EVALUATOR` to the original `evaluate.py` and, for the full-annotation test, `ROXFORD_READABLE_GND` to a readable JSON containing the benchmark's `gnd` and `imlist`. Download the evaluator and original dataset through the [official benchmark page](https://cmp.felk.cvut.cz/revisitop/).

This code collection exposes selected numerical components and existing experimental results. It is not a one-command reproduction of the full image-feature extraction and search pipeline.
