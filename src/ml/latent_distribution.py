import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from applications.utils.latents import get_latents
from ml.utils.load_models import load_default_model, load_trained_model
from data.utils import get_data
from config import DIR_DATA_PROCESSED


def encode_to_latents(model, data, z_dim=8):
    """
    Wrapper to get latents from model and data
    """
    latents = get_latents(
        z=data,
        batch_size=4096,
        use_transformer=False,
        z_dim=z_dim,
        model=model,
    )
    return latents


def plot_latent_distributions(
    data_run_dir,
    model_dir=None,
    method="pca",
    seed=42,
    z_dim=8,
):
    """
    Load data from compile.py outputs and plot latent distributions.
    
    Parameters
    ----------
    data_run_dir : str or Path
        Directory containing the NPZ and JSON files from compile.py
    model_dir: str or Path, optional
        Directory containing the trained model. If None, loads default model from huggingface
    method : str
        One of "pca" or tsne for dimensionality reduction
    seed : int
        Random seed for reproducibility
    z_dim : int
        Number of latent dimensions (default 8)
    model : torch model, optional
        Pre-loaded model. If None, loads default model from config.
    
    Returns
    -------
    dict
        Dictionary containing {
            "fig_train": matplotlib figure for train samples,
            "fig_test": matplotlib figure for test samples,
            "save_path_train": path where train plot will be saved,
            "save_path_test": path where test plot will be saved,
        }
    """
    # load model
    if model_dir is None:
        print("Loading default model...")
        model = load_default_model()
    else:
        print(f"Loading model from {model_dir}...")
        model = load_trained_model(
            model_type="A7X",
            model_dir=model_dir,
        )
    
    data_dir = Path(DIR_DATA_PROCESSED) / data_run_dir
    data, _ = get_data(
        run_dir=data_run_dir, 
        category="all"
    )

    y_train = data["y"][data["train_idx"]]
    y_test = data["y"][data["test_idx"]]

    
    
    # get latents (note latents have shape (n_samples, z_dim + 1) with last column as max)
    latents_train = encode_to_latents(model=model, data=y_train, z_dim=z_dim)
    latents_test = encode_to_latents(model=model, data=y_test, z_dim=z_dim)

    latents_train_viz = latents_train[:, :-1]
    latents_test_viz = latents_test[:, :-1]
    
    # reduce to 2D for visualization
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        proj_train = reducer.fit_transform(latents_train_viz)
        proj_test = reducer.transform(latents_test_viz)
        title_suffix = "PCA"
    elif method.lower() == "tsne":
        # fit on combined data to ensure consistency
        combined = np.vstack([latents_train_viz, latents_test_viz])
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
        proj_combined = reducer.fit_transform(combined)
        proj_train = proj_combined[:len(latents_train_viz)]
        proj_test = proj_combined[len(latents_train_viz):]
        title_suffix = "t-SNE"
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=seed)
        proj_train = reducer.fit_transform(latents_train_viz)
        proj_test = reducer.transform(latents_test_viz)
        title_suffix = "UMAP"
    else:
        raise ValueError(f"Method must be 'pca', 'tsne', or 'umap', got {method}")
    
    # create fig for train samples
    fig_train = plt.figure(figsize=(10, 8))
    plt.scatter(proj_train[:, 0], proj_train[:, 1], alpha=0.1, s=15, c="blue", label="Train")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Train Latent Distribution ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    
    # create fig for test samples
    fig_test = plt.figure(figsize=(10, 8))
    plt.scatter(proj_test[:, 0], proj_test[:, 1], alpha=0.1, s=15, c="red", label="Test")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Test Latent Distribution ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    
    # save paths
    if model_dir is not None:
        model_dir_name = str(model_dir).split('/')[-1]
        figs_dir = data_dir / model_dir_name
    else: 
        figs_dir = data_dir / "old_model"
    
    figs_dir.mkdir(parents=True, exist_ok=True)

    save_path_train = figs_dir / f"latent_dist_train_{method}.png"
    save_path_test = figs_dir / f"latent_dist_test_{method}.png"
    
    return {
        "fig_train": fig_train,
        "fig_test": fig_test,
        "save_path_train": save_path_train,
        "save_path_test": save_path_test,
        "proj_train": proj_train,
        "proj_test": proj_test,
    }


def main(data_run_dir, model_dir=None):
    
    print(f"Processing data from {data_run_dir}")
    data_run_dir = Path(data_run_dir)
    
    for method in ["pca"]:
        print(f"\nGenerating latent distribution plots with {method.upper()}...")
        
        result = plot_latent_distributions(
            data_run_dir=data_run_dir,
            model_dir=model_dir,
            method=method,
            seed=42,
            z_dim=8,
        )
        
        result["fig_train"].savefig(result["save_path_train"], dpi=100, bbox_inches="tight")
        print(f"Saved: {result['save_path_train']}")
        
        result["fig_test"].savefig(result["save_path_test"], dpi=100, bbox_inches="tight")
        print(f"Saved: {result['save_path_test']}")

