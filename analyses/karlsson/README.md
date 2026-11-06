# Karlsson OmniLog dark-condition analysis (revision Figures R2 and R5)

This analysis uses the pretrained A7X-8 model (`you-lab/autoencoder7x-08`) from the parent repository's `src/ml/models/autoencoder7x.py`. It does not retrain the model.

## Setup

Run Jupyter **from the repository root** so `./src` is importable. Install the project's existing dependencies plus `jupyter`, `huggingface_hub`, `scipy`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, and `torch` as needed. Download the six Karlsson OmniLog kinetic CSV files into `analyses/karlsson/data/raw/` (these files are not committed). Model weights must be available from Hugging Face or the local cache.

## Notebook order

1. `Karlsson_dark_preprocessing.ipynb` — parses the raw CSV files and writes `analyses/karlsson/outputs/karlsson_master.pkl` and companion exports. **Note:** this notebook also contains additional exploratory cells after the master-export section; they are preserved from the supplied notebook.
2. `Karlsson_dark_segment_A7X_figR2.ipynb` — splits each 385-point dark trajectory into three 128-point segments (one point dropped), performs A7X inference, and plots latent distributions, correlations, and reconstruction examples.
3. `Karlsson_dark_segment_A7X_figR5.ipynb` — repeats segmentation and inference, and relates eight latent coordinates to five curve-level features using Spearman correlations and binned trends.

R2 and R5 currently **repeat segmentation and model inference**, as in the original analysis. This is intentionally preserved to avoid changing the scientific computation during integration. The two notebooks write some shared intermediate filenames; run sequentially rather than concurrently.

## Reproducibility and limitations

- Paths are relative to the repository root; open Jupyter with that directory as the working directory.
- The supplied random seed is 42; notebook outputs have been cleared for version control.
- Raw data, generated arrays, and figures under `outputs/` are ignored by Git.
