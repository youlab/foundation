import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import json
from pathlib import Path

from config import (
    DIR_RESULTS_ABS_ABUNDANCE_CHAOTIC,
    DIR_RESULTS_ABS_ABUNDANCE_GLV,
)

def load_cv_results(results_dir):
    """Load and process cross-validation results from JSON files.
    
    Args:
        results_dir: Path to directory containing CV result files
        
    Returns:
        Dictionary containing processed metrics with means and standard deviations
    """
    results_dir = Path(results_dir)
    all_raw_r2, all_raw_rmse = [], []
    all_latent_r2, all_latent_rmse = [], []
    training_sizes = None
    
    # Extract model name from directory path
    # Map directory names to file prefixes
    dir_name = results_dir.name
    if "chaotic" in dir_name:
        model_name = "chaotic_mlp"
    elif "glv" in dir_name:
        model_name = "glv_mlp"
    else:
        raise ValueError(f"Unknown directory name: {dir_name}")
    
    # Load results from each CV fold
    for cv in range(5):  # Assuming 5-fold CV
        try:
            with open(results_dir / f"results_{model_name}_cv{cv}.json", 'r') as f:
                data = json.load(f)
            
            # Get training sizes (keys that are numbers)
            train_keys = sorted([int(k) for k in data if k.isdigit()])
            if training_sizes is None:
                training_sizes = train_keys
            
            # Extract metrics
            raw_r2 = [data[str(s)]["raw_accuracy"]["r2"] for s in train_keys]
            raw_rmse = [float(data[str(s)]["raw_accuracy"]["rmse"]) for s in train_keys]
            latent_r2 = [data[str(s)]["latent_accuracy"]["r2"] for s in train_keys]
            latent_rmse = [float(data[str(s)]["latent_accuracy"]["rmse"]) for s in train_keys]
            
            all_raw_r2.append(raw_r2)
            all_raw_rmse.append(raw_rmse)
            all_latent_r2.append(latent_r2)
            all_latent_rmse.append(latent_rmse)
            
        except FileNotFoundError:
            print(f"Warning: Missing file for CV fold {cv}")
            continue
    
    # Convert to numpy arrays
    all_raw_r2 = np.array(all_raw_r2)
    all_raw_rmse = np.array(all_raw_rmse)
    all_latent_r2 = np.array(all_latent_r2)
    all_latent_rmse = np.array(all_latent_rmse)
    
    # Calculate means and standard deviations
    return {
        'training_sizes': np.array(training_sizes),
        'raw_r2_mean': np.mean(all_raw_r2, axis=0),
        'raw_r2_std': np.std(all_raw_r2, axis=0),
        'latent_r2_mean': np.mean(all_latent_r2, axis=0),
        'latent_r2_std': np.std(all_latent_r2, axis=0),
        'raw_rmse_mean': np.mean(all_raw_rmse, axis=0),
        'raw_rmse_std': np.std(all_raw_rmse, axis=0),
        'latent_rmse_mean': np.mean(all_latent_rmse, axis=0),
        'latent_rmse_std': np.std(all_latent_rmse, axis=0),
    }

def plot_absolute_abundance_metrics(
    ax_chaotic_r2,
    ax_chaotic_rmse,
    ax_glv_r2,
    ax_glv_rmse,
    fs_label=24,
    fs_ticks=18,
    fs_legend=14,
):
    """Plot R² and RMSE metrics for absolute abundance prediction.
    
    Args:
        ax_chaotic_r2: Matplotlib axis for chaotic R² plot
        ax_chaotic_rmse: Matplotlib axis for chaotic RMSE plot
        ax_glv_r2: Matplotlib axis for GLV R² plot
        ax_glv_rmse: Matplotlib axis for GLV RMSE plot
        fs_label: Font size for axis labels 20
        fs_ticks: Font size for tick labels 16
        fs_legend: Font size for legend
    """
    # Load and process results for both datasets
    metrics_chaotic = load_cv_results(DIR_RESULTS_ABS_ABUNDANCE_CHAOTIC)
    metrics_glv = load_cv_results(DIR_RESULTS_ABS_ABUNDANCE_GLV)
    
    # Plot Chaotic R² scores
    ax_chaotic_r2.plot(
        metrics_chaotic['training_sizes'],
        metrics_chaotic['raw_r2_mean'],
        'o-',
        color='tab:orange',
        label='Using raw',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    ax_chaotic_r2.plot(
        metrics_chaotic['training_sizes'],
        metrics_chaotic['latent_r2_mean'],
        's-',
        color='tab:blue',
        label='Using latent',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    
    # Plot Chaotic RMSE
    ax_chaotic_rmse.plot(
        metrics_chaotic['training_sizes'],
        metrics_chaotic['raw_rmse_mean'],
        'o-',
        color='tab:orange',
        label='Using raw',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    ax_chaotic_rmse.plot(
        metrics_chaotic['training_sizes'],
        metrics_chaotic['latent_rmse_mean'],
        's-',
        color='tab:blue',
        label='Using latent',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    
    # Plot GLV R² scores
    ax_glv_r2.plot(
        metrics_glv['training_sizes'],
        metrics_glv['raw_r2_mean'],
        'o-',
        color='tab:orange',
        label='Using raw',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    ax_glv_r2.plot(
        metrics_glv['training_sizes'],
        metrics_glv['latent_r2_mean'],
        's-',
        color='tab:blue',
        label='Using latent',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    
    # Plot GLV RMSE
    ax_glv_rmse.plot(
        metrics_glv['training_sizes'],
        metrics_glv['raw_rmse_mean'],
        'o-',
        color='tab:orange',
        label='Using raw',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    ax_glv_rmse.plot(
        metrics_glv['training_sizes'],
        metrics_glv['latent_rmse_mean'],
        's-',
        color='tab:blue',
        label='Using latent',
        lw=2,
        markersize=5,
        alpha=0.8,
    )
    
    # Style all plots
    all_axes = [ax_chaotic_r2, ax_chaotic_rmse, ax_glv_r2, ax_glv_rmse]
    
    # Style Chaotic R² plot - LARGER FONTS, ONLY ONE LEGEND, NO X-LABEL
    # ax_chaotic_r2.set_xlabel('Train Size', fontsize=fs_label+2, labelpad=8, fontweight='bold')  # REMOVED for decluttering
    ax_chaotic_r2.set_ylabel('R²', fontsize=fs_label+2)
    ax_chaotic_r2.set_title('GLV with dispersal', fontsize=fs_label+4, pad=25)  # INCREASED pad from 20 to 25 for consistency
    ax_chaotic_r2.tick_params(axis='both', which='major', labelsize=fs_ticks+2)
    ax_chaotic_r2.spines['top'].set_visible(False)
    ax_chaotic_r2.spines['right'].set_visible(False)
    ax_chaotic_r2.legend(fontsize=fs_legend+2, loc='lower right', frameon=False)  # Keep legend in panel C
    
    # Style Chaotic RMSE plot - NO LEGEND, NO X-LABEL
    # ax_chaotic_rmse.set_xlabel('Train Size', fontsize=fs_label+2, labelpad=8, fontweight='bold')  # REMOVED for decluttering
    ax_chaotic_rmse.set_ylabel('RMSE', fontsize=fs_label+2)
    ax_chaotic_rmse.tick_params(axis='both', which='major', labelsize=fs_ticks+2)
    ax_chaotic_rmse.spines['top'].set_visible(False)
    ax_chaotic_rmse.spines['right'].set_visible(False)
    # NO LEGEND for panel D
    
    # Style GLV R² plot - NO LEGEND, HAS X-LABEL
    ax_glv_r2.set_xlabel('Train size', fontsize=fs_label+2, labelpad=6)  # KEPT in E
    ax_glv_r2.set_ylabel('R²', fontsize=fs_label+2)
    ax_glv_r2.set_title('GLV without dispersal', fontsize=fs_label+4, pad=15)  # INCREASED pad from 20 to 25 to avoid overlap with tick labels
    ax_glv_r2.tick_params(axis='both', which='major', labelsize=fs_ticks+2)
    ax_glv_r2.spines['top'].set_visible(False)
    ax_glv_r2.spines['right'].set_visible(False)
    # NO LEGEND for panel E
    
    # Style GLV RMSE plot - NO LEGEND, HAS X-LABEL
    ax_glv_rmse.set_xlabel('Train size', fontsize=fs_label+2, labelpad=6)  # KEPT in F
    ax_glv_rmse.set_ylabel('RMSE', fontsize=fs_label+2)
    ax_glv_rmse.tick_params(axis='both', which='major', labelsize=fs_ticks+2)
    ax_glv_rmse.spines['top'].set_visible(False)
    ax_glv_rmse.spines['right'].set_visible(False)
    # NO LEGEND for panel F
    
    # Set axis limits and ticks for R² plots - DECLUTTERED (only 3 tick labels)
    for ax in [ax_chaotic_r2, ax_glv_r2]:
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks([0.2, 0.6, 1.0])  # Only 3 ticks for decluttering
        ax.set_yticklabels(['0.2', '0.6', '1.0'])
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))  # Adjusted for fewer major ticks
    
    # Set axis limits and ticks for RMSE plots - DECLUTTERED (only 3 tick labels)
    for ax in [ax_chaotic_rmse, ax_glv_rmse]:
        ax.set_ylim(0, 0.5)  # More appropriate range for RMSE
        ax.set_yticks([0, 0.3, 0.5])  # Only 3 ticks for decluttering
        ax.set_yticklabels(['0', '0.3', '0.5'])
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    
    # Set x-axis ticks (fewer major ticks like figure 3)
    max_train_chaotic = metrics_chaotic['training_sizes'].max()
    max_train_glv = metrics_glv['training_sizes'].max()
    
    ax_chaotic_r2.set_xticks([0, max_train_chaotic/2, max_train_chaotic])
    ax_chaotic_rmse.set_xticks([0, max_train_chaotic/2, max_train_chaotic])
    ax_glv_r2.set_xticks([0, max_train_glv/2, max_train_glv])
    ax_glv_rmse.set_xticks([0, max_train_glv/2, max_train_glv])
    
    # Format x-axis labels in thousands (K) instead of scientific notation
    ax_chaotic_r2.set_xticklabels(['0', f'{int(max_train_chaotic/2000)}K', f'{int(max_train_chaotic/1000)}K'])
    ax_chaotic_rmse.set_xticklabels(['0', f'{int(max_train_chaotic/2000)}K', f'{int(max_train_chaotic/1000)}K'])
    ax_glv_r2.set_xticklabels(['0', f'{int(max_train_glv/2000)}K', f'{int(max_train_glv/1000)}K'])
    ax_glv_rmse.set_xticklabels(['0', f'{int(max_train_glv/2000)}K', f'{int(max_train_glv/1000)}K'])
    
    # Add minor ticks
    for ax in all_axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Fewer minor ticks

