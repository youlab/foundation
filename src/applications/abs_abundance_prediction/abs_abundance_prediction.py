import logging
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from applications.antibiotics.generate_results import main as train_main
from applications.utils.latents import get_latents
from ml.utils.load_models import load_model
from config import SEQ_LEN, Z_DIM, DIR_RESULTS_ABS_ABUNDANCE


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim))
    def forward(self, x):
        return self.net(x)
    

def setup_logger(log_file=None):
    """Initialize and return a logger."""
    logger = logging.getLogger("absolute_abundance_prediction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_file}")
    return logger

logger = setup_logger()


def load_abundance_from_npz(relative_npz_path: str, absolute_npz_path: str):
    """Load relative and absolute abundance tensors from provided .npz files.
    
    Simplified version that loads specific keys directly.

    Returns
    -------
    focal: np.ndarray
        Absolute abundances with shape (n_samples, n_species, n_time) 
    rel: np.ndarray
        Relative abundances with shape (n_samples, n_species, n_time)
    """
    logger.info(f"Loading relative abundance from: {relative_npz_path}")
    rel_data = np.load(relative_npz_path)
    rel = rel_data["relative_abundance"]
    
    logger.info(f"Loading total abundance from: {absolute_npz_path}")
    total_data = np.load(absolute_npz_path)
    total = total_data["total_abundance"]  # Shape: (n_samples, n_time)
    
    logger.info(f"Loaded rel shape: {rel.shape}, total shape: {total.shape}")
    
    n_samples, n_species, n_time = rel.shape
    focal = np.expand_dims(total, axis=1)  # Shape: (n_samples, 1, n_time)
    focal = focal / n_species
    focal = np.tile(focal, (1, n_species, 1))  # Shape: (n_samples, n_species, n_time)
    
    logger.info(f"Final shapes → focal: {focal.shape}, rel: {rel.shape}")
    return focal, rel

def train_single_mlp(X_train, X_test, Y_train, Y_test, epochs=100, lr=0.001, patience=10, min_delta=1e-6):
    """Train a single MLP model with early stopping and return predictions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )
    
    X_train_tensor = torch.FloatTensor(X_train_split).to(device)
    X_val_tensor = torch.FloatTensor(X_val_split).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device) 
    Y_train_tensor = torch.FloatTensor(Y_train_split).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val_split).to(device)
    Y_test_tensor = torch.FloatTensor(Y_test).to(device)
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = MLP(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, Y_val_tensor).item()
        model.train()
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Log progress occasionally
        if (epoch + 1) % 20 == 0:
            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(f"MLP Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
            break
    
    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model weights")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        Y_test_pred = model(X_test_tensor).cpu().numpy()
    
    return Y_test_pred


def train_mlp_with_cross_validation(X, Y, cross_val=0, train_sizes=None, epochs=100, lr=0.001, model_type="A7X", z_dim=Z_DIM, patience=10, min_delta=1e-6):
    """Train MLP with proper cross-validation structure matching ExtraTree implementation."""
    if train_sizes is None:
        train_sizes = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # train_sizes = [0.7, 0.8]

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training MLP on device: {device} for cross_val={cross_val}")
    
    # Load VAE model (same as ExtraTree implementation)
    logger.info("Loading VAE model for latent conversion...")
    vae_model = load_model(model_type=model_type, z_dim=z_dim)
    logger.info(f"VAE model loaded: {type(vae_model)}")
    
    # Convert X to latent representations using VAE (same as ExtraTree)
    logger.info(f"Converting input X to latents...")
    logger.info(f"Reshaping X from {X.shape} to {X.reshape(-1, SEQ_LEN).shape} for VAE")
    X_latents = get_latents(
        z=X.reshape(-1, SEQ_LEN),
        use_transformer=model_type == "MCR",
        z_dim=z_dim,
        model=vae_model,
    ).reshape(X.shape[0], -1)
    logger.info(f"X_latents extracted - shape: {X_latents.shape}")
    logger.info(f"X_latents range: [{np.min(X_latents):.4f}, {np.max(X_latents):.4f}]")
    
    np.random.seed(42 + cross_val)  # Ensure reproducible splits
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(0.2 * len(X))
    test_indices = indices[:test_size]
    train_indices_full = indices[test_size:]
    
    # Split both raw and latent data
    X_test = X[test_indices]
    X_latents_test = X_latents[test_indices]
    Y_test = Y[test_indices]
    X_train_full = X[train_indices_full]
    X_latents_train_full = X_latents[train_indices_full]
    Y_train_full = Y[train_indices_full]
    
    results = {}
    viz_samples = None
    
    for train_size in train_sizes:
        logger.info(f"Training MLP with train_size: {train_size}")
        
        # Determine actual number of training samples
        if train_size == 0.8:  # Use full training set
            X_train = X_train_full
            X_latents_train = X_latents_train_full
            Y_train = Y_train_full
        else:
            # Sample from the training set
            n_train_samples = int(train_size * len(X_train_full) / 0.8)
            if n_train_samples < 1:
                n_train_samples = 1
            if n_train_samples > len(X_train_full):
                n_train_samples = len(X_train_full)
            
            train_sample_indices = np.random.choice(len(X_train_full), n_train_samples, replace=False)
            X_train = X_train_full[train_sample_indices]
            X_latents_train = X_latents_train_full[train_sample_indices]
            Y_train = Y_train_full[train_sample_indices]
        
        logger.info(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        logger.info(f"Raw input dim: {X_train.shape[1]}, Latent input dim: {X_latents_train.shape[1]}")
        
        # Train MLP on RAW data
        logger.info("Training MLP on RAW data...")
        Y_test_pred_raw = train_single_mlp(X_train, X_test, Y_train, Y_test, epochs=epochs, lr=lr, patience=patience, min_delta=min_delta)
        
        # Train MLP on LATENT data  
        logger.info("Training MLP on LATENT data...")
        Y_test_pred_latent = train_single_mlp(X_latents_train, X_latents_test, Y_train, Y_test, epochs=epochs, lr=lr, patience=patience, min_delta=min_delta)
        
        # Calculate metrics for both approaches
        raw_r2 = r2_score(Y_test.flatten(), Y_test_pred_raw.flatten())
        raw_rmse = np.sqrt(mean_squared_error(Y_test.flatten(), Y_test_pred_raw.flatten()))
        latent_r2 = r2_score(Y_test.flatten(), Y_test_pred_latent.flatten())
        latent_rmse = np.sqrt(mean_squared_error(Y_test.flatten(), Y_test_pred_latent.flatten()))
        
        results[X_train.shape[0]] = {
            'raw_accuracy': {'r2': raw_r2, 'rmse': raw_rmse},
            'latent_accuracy': {'r2': latent_r2, 'rmse': latent_rmse}
        }
        
        # Save visualization samples for the largest training set (store 5 examples)
        if train_size == train_sizes[-1]:
            k = min(5, len(Y_test))
            viz_samples = {
                'indices': np.arange(k),
                'true_values': Y_test[:k],
                'raw_predictions': Y_test_pred_raw[:k],
                'latent_predictions': Y_test_pred_latent[:k],
            }
        
        logger.info(f"MLP Train size {X_train.shape[0]} - Raw: R2={raw_r2:.4f}, RMSE={raw_rmse:.4f} | Latent: R2={latent_r2:.4f}, RMSE={latent_rmse:.4f}")
    
    # Add test size and visualization samples
    results['test_size'] = X_test.shape[0]
    if viz_samples:
        results['viz_samples'] = viz_samples
    
    return results


def train_model(rel, focal, use_mlp=False, mlp_epochs=100, mlp_lr=0.001, mlp_patience=10, mlp_min_delta=1e-6):
    """Prepare features and train using either ExtraTree (default) or MLP."""

    n_samples, n_species, n_time = rel.shape

    index_mapping = []
    X, Y = [], []
    
    # Option 1: Use complete trajectories (no sliding window)
    for s in range(n_samples):
        # Use the entire trajectory as input
        x = rel[s, :, :SEQ_LEN]  # Take first SEQ_LEN time points
        # Get total abundance for the same time window
        total_abundance = focal[s, :, :SEQ_LEN].sum(axis=0)  # Sum across species
        X.append(x.flatten())
        Y.append(total_abundance)
        index_mapping.append((s, 0))  # Always start from time 0
    
    X = np.nan_to_num(np.array(X))
    Y = np.nan_to_num(np.array(Y))

    model_type = "MLP" if use_mlp else "ExtraTree"
    logger.info(f"Training data shapes - X: {X.shape}, Y: {Y.shape}")
    logger.info(f"X represents relative abundances for {n_species} species")
    logger.info(f"Y represents total abundance (sum across all species)")
    logger.info(f"Using complete trajectories starting from t=0 (no sliding window)")
    logger.info(f"Model type: {model_type}")

    data_context = {
        'rel': rel,
        'focal': focal,
        'index_mapping': index_mapping,
        'n_species': n_species
    }

    train_sizes = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # train_sizes = [0.7, 0.8]

    results = []
    
    if use_mlp:
        # Use MLP training with proper cross-validation
        logger.info("Training with MLP...")
        for cross_val in range(5):
            res = train_mlp_with_cross_validation(
                X, Y, 
                cross_val=cross_val,
                train_sizes=train_sizes,
                epochs=mlp_epochs, 
                lr=mlp_lr,
                model_type="A7X",
                z_dim=Z_DIM,
                patience=mlp_patience,
                min_delta=mlp_min_delta
            )
            res['data_context'] = data_context
            results.append(res)
    else:
        # Use existing ExtraTree training
        logger.info("Training with ExtraTree...")
        for cross_val in range(5):
            res = train_main(
                x_raw=X,
                tgt=Y,
                model_type="A7X",
                z_dim=Z_DIM,
                classify=False,
                detailed_binary=False,
                epochs_fine_tuned=0,
                lr_fine_tuned=0.0,
                epochs_end2end=0,
                lr_end2end=0.0,
                return_latent_vectors=False,
                stack=1,  # Changed from n_species to 1 since we're predicting a single value
                cross_val=cross_val,
                max_depth=None,
                train_sizes=train_sizes,
                name_suffix="glv_analysis",
                use_decoded_prediction=False,
                use_tgt_latent=False,
                include_rmse=True,
                save_samples_for_viz=True,
                n_viz_samples=5,
            )
            # Add data context for visualization
            res['data_context'] = data_context
            results.append(res)
    
    return results

def visualize_grid_5x5(results, output_dir=None, prediction_key: str = "raw_predictions"):
    """
    Create a 5x5 grid visualization: 5 CV folds x 5 samples.
    Each cell shows true vs predicted total abundance over the first SEQ_LEN time points.

    Args:
        results: List of per-fold results (expects 'viz_samples' to contain 'true_values' and predictions).
        output_dir: Directory to save the grid image. If None, shows the figure.
        prediction_key: Which predictions to plot ('raw_predictions' or 'latent_predictions').
    """
    num_rows = min(5, len(results))
    if num_rows == 0:
        logger.warning("No results provided for grid visualization.")
        return

    num_cols = 5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.2, num_rows * 2.6), sharex=True, sharey=False)
    # Ensure axes is 2D array
    if num_rows == 1:
        axes = np.array([axes])

    time_window = np.arange(SEQ_LEN)

    for row_idx in range(num_rows):
        res = results[row_idx]
        viz_samples = res.get("viz_samples", None)
        if viz_samples is None:
            logger.warning(f"Result {row_idx} has no 'viz_samples'; skipping row.")
            continue

        true_values = viz_samples.get("true_values", None)
        predictions = viz_samples.get(prediction_key, None)
        if true_values is None or predictions is None:
            logger.warning(f"Result {row_idx} missing required keys for visualization; skipping row.")
            continue

        k = min(num_cols, len(true_values))
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.grid(alpha=0.2)
            if col_idx < k:
                y_true = np.asarray(true_values[col_idx])
                y_pred = np.asarray(predictions[col_idx])
                # Clip to SEQ_LEN if longer
                y_true = y_true[: len(time_window)]
                y_pred = y_pred[: len(time_window)]
                ax.plot(time_window[: len(y_true)], y_true, 'k-', linewidth=1.5, label='True')
                ax.plot(time_window[: len(y_pred)], y_pred, 'r--', linewidth=1.5, label='Pred')
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx+1}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"CV {row_idx+1}")

    # Add one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle(f"True vs Predicted Total Abundance (prediction: {prediction_key})", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / "grid_5x5.png"
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        logger.info(f"Saved 5x5 grid visualization to: {save_path}")
        plt.close()
    else:
        plt.show()
    return


def main(output_dir=None, mlp_epochs=100, mlp_lr=0.001, mlp_patience=10, mlp_min_delta=1e-6,
         relative_npz: str = None, absolute_npz: str = None, dataset_label: str = None):
    """Main entry point.

    If `relative_npz` and `absolute_npz` are provided, loads pre-saved data from these files.
    """
    
    # Decide dataset label for output naming
    if dataset_label is None:
        inferred_label = "glv"
        for token in [relative_npz, absolute_npz]:
            if token and isinstance(token, str) and "chaotic" in token.lower():
                inferred_label = "chaotic"
        dataset_label = inferred_label

    if output_dir is None:
        model_suffix = "mlp" 
        output_dir = DIR_RESULTS_ABS_ABUNDANCE / f'{dataset_label}_analysis_results_{model_suffix}'
    
    output_dir = Path(output_dir)
    
    model_name = "MLP"

    # Load data from provided NPZ files
    if not (relative_npz and absolute_npz):
        raise ValueError("Both --relative_npz and --absolute_npz must be provided.")
    logger.info(f"Loading dataset '{dataset_label}' from NPZ files with {model_name}...")
    focal, rel = load_abundance_from_npz(relative_npz, absolute_npz)
    
    if focal is None or rel is None:
        return None

    # Basic validation for sequence length
    if rel.shape[2] < SEQ_LEN:
        raise ValueError(
            f"Time dimension ({rel.shape[2]}) is shorter than SEQ_LEN ({SEQ_LEN}). Please provide data with at least {SEQ_LEN} time points."
        )
        
    results = train_model(rel, focal, use_mlp=True, mlp_epochs=mlp_epochs, mlp_lr=mlp_lr, mlp_patience=mlp_patience, mlp_min_delta=mlp_min_delta)
    
    # Save results and visualizations
    for i, res in enumerate(results):
        model_type = "mlp"
        json_path = output_dir / f"results_{dataset_label}_{model_type}_cv{i}.json"
        json_path.parent.mkdir(exist_ok=True, parents=True)
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"Saved results to: {json_path}")
    
    viz_dir = output_dir / "visualizations"
    logger.info(f"Creating visualization directory at: {viz_dir}")
    # visualize_curves(results, output_dir=viz_dir)
    # 5x5 grid across CV folds (rows) and 5 samples (cols)
    visualize_grid_5x5(results, output_dir=viz_dir, prediction_key="raw_predictions")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Absolute abundance prediction using MLP. Load from .npz files.')
    
    # Data loading parameters (required)
    parser.add_argument('--relative_npz', type=str, required=True, help='Path to relative abundance .npz file')
    parser.add_argument('--absolute_npz', type=str, required=True, help='Path to absolute abundance .npz file')
    parser.add_argument('--dataset_label', type=str, default=None, choices=["glv", "chaotic"], help='Dataset label used in outputs (auto-inferred if not provided)')

    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    parser.add_argument('--mlp_epochs', type=int, default=100, 
                       help='Number of epochs for MLP training (default: 100)')
    parser.add_argument('--mlp_lr', type=float, default=0.001, 
                       help='Learning rate for MLP training (default: 0.001)')
    parser.add_argument('--mlp_patience', type=int, default=10, 
                       help='Early stopping patience for MLP training (default: 10)')
    parser.add_argument('--mlp_min_delta', type=float, default=1e-6, 
                       help='Minimum change for early stopping (default: 1e-6)')
    
    args = parser.parse_args()
    
    results = main(
        output_dir=args.output_dir,
        mlp_epochs=args.mlp_epochs,
        mlp_lr=args.mlp_lr,
        mlp_patience=args.mlp_patience,
        mlp_min_delta=args.mlp_min_delta,
        relative_npz=args.relative_npz,
        absolute_npz=args.absolute_npz,
        dataset_label=args.dataset_label
    )
    
    if results:
        model_type = "mlp"
        logger.info(f"Absolute abundance prediction analysis with {model_type} completed successfully.") 