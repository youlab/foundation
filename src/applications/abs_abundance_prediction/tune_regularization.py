"""
Hyperparameter tuning script for L2 regularization.

This script systematically tests different L2 regularization combinations for raw vs latent features
and creates comprehensive visualizations comparing their performance.

Usage:
    python tune_regularization.py \
        --relative_npz <path> \
        --absolute_npz <path> \
        --output_dir <output_dir> \
        --n_trials 5 \
        --dataset_label glv
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from itertools import product

from applications.abs_abundance_prediction.abs_abundance_prediction import (
    main, 
    load_abundance_from_npz
)
from config import SEQ_LEN, Z_DIM, DIR_RESULTS_ABS_ABUNDANCE


def setup_logger(log_file=None):
    """Initialize and return a logger."""
    logger = logging.getLogger("regularization_tuner")
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
    return logger


logger = setup_logger()


def tune_regularization(relative_npz, absolute_npz, output_dir, dataset_label="glv", n_trials=5):
    """
    Systematically test different L2 regularization combinations.
    
    Args:
        relative_npz: Path to relative abundance NPZ file
        absolute_npz: Path to absolute abundance NPZ file
        output_dir: Output directory
        dataset_label: Dataset label
        n_trials: Number of trials (CV folds) per configuration
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define L2 regularization grid
    # Higher values: 1e-1, 1e-2, 1e-3
    # Medium values: 5e-4, 1e-4
    # Lower values: 1e-5, 0 (no reg)
    l2_values = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    logger.info("=" * 80)
    logger.info("REGULARIZATION HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"L2 Regularization values to test: {l2_values}")
    logger.info(f"Will test {len(l2_values)**2} combinations (raw × latent)")
    logger.info(f"Each configuration will use {n_trials} trials (CV folds)")
    
    # Store all results
    tuning_results = {}
    
    config_idx = 0
    total_configs = len(l2_values) ** 2
    
    for l2_raw, l2_latent in product(l2_values, l2_values):
        config_idx += 1
        logger.info("\n" + "=" * 80)
        logger.info(f"Configuration {config_idx}/{total_configs}: L2_raw={l2_raw:.0e}, L2_latent={l2_latent:.0e}")
        logger.info("=" * 80)
        
        config_key = f"raw_{l2_raw:.0e}_latent_{l2_latent:.0e}"
        config_output_dir = output_dir / config_key
        
        try:
            results = main(
                output_dir=config_output_dir,
                mlp_epochs=100,
                mlp_lr=0.001,
                mlp_patience=10,
                mlp_min_delta=1e-6,
                relative_npz=relative_npz,
                absolute_npz=absolute_npz,
                dataset_label=dataset_label,
                mlp_l2_raw=l2_raw,
                mlp_l2_latent=l2_latent
            )
            
            if results is None:
                logger.warning(f"Skipping config due to error")
                continue
            
            # Extract aggregate metrics
            train_sizes = []
            raw_r2_list = []
            latent_r2_list = []
            
            for cv_fold in results:
                for train_size_str, metrics in cv_fold.items():
                    if train_size_str in ['test_size', 'data_context', 'viz_samples']:
                        continue
                    try:
                        train_size = int(train_size_str)
                    except (ValueError, TypeError):
                        continue
                    
                    if 'raw_accuracy' in metrics:
                        train_sizes.append(train_size)
                        raw_r2_list.append(metrics['raw_accuracy']['r2'])
                        latent_r2_list.append(metrics['latent_accuracy']['r2'])
            
            if not raw_r2_list:
                logger.warning(f"No metrics extracted for this config")
                continue
            
            # Compute aggregate statistics
            avg_raw_r2 = np.mean(raw_r2_list)
            avg_latent_r2 = np.mean(latent_r2_list)
            std_raw_r2 = np.std(raw_r2_list)
            std_latent_r2 = np.std(latent_r2_list)
            
            # Check trend (is raw declining with more data?)
            unique_sizes = sorted(set(train_sizes))
            raw_r2_by_size = [np.mean([raw_r2_list[i] for i in range(len(train_sizes)) if train_sizes[i] == s]) 
                              for s in unique_sizes]
            raw_r2_trend = np.polyfit(np.arange(len(raw_r2_by_size)), raw_r2_by_size, 1)[0]
            
            tuning_results[config_key] = {
                'l2_raw': float(l2_raw),
                'l2_latent': float(l2_latent),
                'avg_raw_r2': float(avg_raw_r2),
                'avg_latent_r2': float(avg_latent_r2),
                'std_raw_r2': float(std_raw_r2),
                'std_latent_r2': float(std_latent_r2),
                'raw_r2_trend': float(raw_r2_trend),
                'latent_advantage': float(avg_latent_r2 - avg_raw_r2),
            }
            
            logger.info(f"✓ Avg R² - Raw: {avg_raw_r2:.4f}±{std_raw_r2:.4f}, Latent: {avg_latent_r2:.4f}±{std_latent_r2:.4f}")
            logger.info(f"  Raw R² trend: {raw_r2_trend:+.6f} (negative = declining with more data)")
            logger.info(f"  Latent advantage: {avg_latent_r2 - avg_raw_r2:+.4f}")
            
        except Exception as e:
            logger.error(f"✗ Error in config: {str(e)}")
            continue
    
    # Save tuning results
    tuning_path = output_dir / "tuning_results.json"
    with open(tuning_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    logger.info(f"\nSaved tuning results to: {tuning_path}")
    
    # Visualize tuning results
    visualize_tuning_results(tuning_results, output_dir, l2_values)
    
    return tuning_results


def visualize_tuning_results(tuning_results, output_dir, l2_values):
    """Create comprehensive visualizations of tuning results."""
    
    output_dir = Path(output_dir)
    
    if not tuning_results:
        logger.warning("No tuning results to visualize")
        return
    
    logger.info("\nGenerating tuning visualizations...")
    
    # Prepare data for heatmaps
    l2_values_sorted = sorted(set(l2_values))
    n_vals = len(l2_values_sorted)
    
    raw_r2_heatmap = np.zeros((n_vals, n_vals))
    latent_r2_heatmap = np.zeros((n_vals, n_vals))
    latent_advantage_heatmap = np.zeros((n_vals, n_vals))
    raw_trend_heatmap = np.zeros((n_vals, n_vals))
    
    # Fill heatmaps
    for result in tuning_results.values():
        l2_raw = result['l2_raw']
        l2_latent = result['l2_latent']
        
        i = l2_values_sorted.index(l2_raw)
        j = l2_values_sorted.index(l2_latent)
        
        raw_r2_heatmap[i, j] = result['avg_raw_r2']
        latent_r2_heatmap[i, j] = result['avg_latent_r2']
        latent_advantage_heatmap[i, j] = result['latent_advantage']
        raw_trend_heatmap[i, j] = result['raw_r2_trend']
    
    # Format labels
    label_strs = [f"{v:.0e}" for v in l2_values_sorted]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Raw R² heatmap
    im0 = axes[0, 0].imshow(raw_r2_heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_xticks(range(len(label_strs)))
    axes[0, 0].set_yticks(range(len(label_strs)))
    axes[0, 0].set_xticklabels(label_strs, rotation=45, ha='right')
    axes[0, 0].set_yticklabels(label_strs)
    axes[0, 0].set_xlabel('L2 Regularization (Latent Features)')
    axes[0, 0].set_ylabel('L2 Regularization (Raw Features)')
    axes[0, 0].set_title('Average R² - Raw Features')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Latent R² heatmap
    im1 = axes[0, 1].imshow(latent_r2_heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_xticks(range(len(label_strs)))
    axes[0, 1].set_yticks(range(len(label_strs)))
    axes[0, 1].set_xticklabels(label_strs, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(label_strs)
    axes[0, 1].set_xlabel('L2 Regularization (Latent Features)')
    axes[0, 1].set_ylabel('L2 Regularization (Raw Features)')
    axes[0, 1].set_title('Average R² - Latent Features')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Latent Advantage heatmap
    im2 = axes[1, 0].imshow(latent_advantage_heatmap, cmap='RdBu_r', aspect='auto', 
                            vmin=-0.2, vmax=0.2)
    axes[1, 0].set_xticks(range(len(label_strs)))
    axes[1, 0].set_yticks(range(len(label_strs)))
    axes[1, 0].set_xticklabels(label_strs, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(label_strs)
    axes[1, 0].set_xlabel('L2 Regularization (Latent Features)')
    axes[1, 0].set_ylabel('L2 Regularization (Raw Features)')
    axes[1, 0].set_title('Latent R² - Raw R² (Green=Raw better, Red=Latent better)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Raw R² Trend heatmap
    im3 = axes[1, 1].imshow(raw_trend_heatmap, cmap='RdYlGn_r', aspect='auto', 
                            vmin=-0.01, vmax=0.01)
    axes[1, 1].set_xticks(range(len(label_strs)))
    axes[1, 1].set_yticks(range(len(label_strs)))
    axes[1, 1].set_xticklabels(label_strs, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(label_strs)
    axes[1, 1].set_xlabel('L2 Regularization (Latent Features)')
    axes[1, 1].set_ylabel('L2 Regularization (Raw Features)')
    axes[1, 1].set_title('Raw R² Trend (Green=stable/improving, Red=declining)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    save_path = output_dir / "tuning_heatmaps.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved tuning heatmaps to: {save_path}")
    plt.close()
    
    # Create recommendations figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best configurations
    sorted_results = sorted(tuning_results.items(), 
                           key=lambda x: x[1]['avg_latent_r2'], 
                           reverse=True)
    
    # Top 10 configurations
    top_k = min(10, len(sorted_results))
    top_configs = sorted_results[:top_k]
    
    config_names = [k.replace('raw_', 'R:').replace('latent_', 'L:').replace('_', ' ') 
                    for k, _ in top_configs]
    raw_scores = [v['avg_raw_r2'] for _, v in top_configs]
    latent_scores = [v['avg_latent_r2'] for _, v in top_configs]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    ax.barh(x - width/2, raw_scores, width, label='Raw', alpha=0.8)
    ax.barh(x + width/2, latent_scores, width, label='Latent', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(config_names, fontsize=9)
    ax.set_xlabel('Average R² Score')
    ax.set_title('Top 10 Regularization Configurations')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = output_dir / "tuning_recommendations.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved tuning recommendations to: {save_path}")
    plt.close()
    
    # Print recommendations
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 80)
    for i, (config_key, result) in enumerate(top_configs, 1):
        logger.info(f"\n{i}. {config_key}")
        logger.info(f"   Raw L²: {result['l2_raw']:.0e}, Latent L²: {result['l2_latent']:.0e}")
        logger.info(f"   Raw R²:    {result['avg_raw_r2']:.4f} ± {result['std_raw_r2']:.4f}")
        logger.info(f"   Latent R²: {result['avg_latent_r2']:.4f} ± {result['std_latent_r2']:.4f}")
        logger.info(f"   Latent advantage: {result['latent_advantage']:+.4f}")
        logger.info(f"   Raw trend: {result['raw_r2_trend']:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tune L2 regularization parameters for raw vs latent features.'
    )
    
    parser.add_argument('--relative_npz', type=str, required=True,
                       help='Path to relative abundance .npz file')
    parser.add_argument('--absolute_npz', type=str, required=True,
                       help='Path to absolute abundance .npz file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for tuning results')
    parser.add_argument('--dataset_label', type=str, default='glv',
                       choices=['glv', 'chaotic'],
                       help='Dataset label')
    parser.add_argument('--n_trials', type=int, default=5,
                       help='Number of CV folds to run per configuration (default: 5)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = DIR_RESULTS_ABS_ABUNDANCE / f'{args.dataset_label}_regularization_tuning'
    
    logger.info(f"Starting regularization tuning for dataset: {args.dataset_label}")
    logger.info(f"Output directory: {args.output_dir}")
    
    tune_regularization(
        relative_npz=args.relative_npz,
        absolute_npz=args.absolute_npz,
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        n_trials=args.n_trials
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Regularization tuning complete!")
    logger.info("=" * 80)
