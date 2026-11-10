# Strain-level resistance results

Five deterministic GroupKFold folds; largest training size; curve-level binary F1.
Values are mean ± population standard deviation across folds.

| Antibiotic | Raw F1 | A7X latent F1 | Resistant isolates |
| --- | --- | --- | --- |
| SAM | 0.672 ± 0.020 | 0.698 ± 0.037 | 158/277 |
| GM | 0.143 ± 0.042 | 0.138 ± 0.031 | 50/277 |
| SXT | 0.605 ± 0.008 | 0.621 ± 0.012 | 155/277 |
| CIP | 0.696 ± 0.026 | 0.702 ± 0.023 | 163/277 |

All 4,432 curves from 277 isolates were checked for consistent labels.
Every isolate stayed in one supervised fold, with zero train/test strain overlap.

Reproduce from `src` using the command in the repository README.
