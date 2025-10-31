"""
Simple script to plot R² and RMSE for all L2 configurations.

Creates clear bar charts showing performance metrics for each L2 combination,
with filenames indicating the configuration.

Usage:
    python plot_tuning_metrics.py \
        --tuning_results results/chaotic_regularization_tuning/tuning_results.json \
        --output_dir results/chaotic_regularization_tuning/metric_plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging


def setup_logger():
    """Initialize and return a logger."""
    logger = logging.getLogger("metric_plotter")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def plot_r2_metrics(tuning_results: dict, output_dir: Path, dataset_label: str = ""):
    """Plot R² scores for all configurations."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    configs = []
    for config_key, metrics in tuning_results.items():
        configs.append({
            'key': config_key,
            'label': f"R:{metrics['l2_raw']:.0e} L:{metrics['l2_latent']:.0e}",
            'l2_raw': metrics['l2_raw'],
            'l2_latent': metrics['l2_latent'],
            'raw_r2': metrics['avg_raw_r2'],
            'latent_r2': metrics['avg_latent_r2'],
            'raw_r2_std': metrics['std_raw_r2'],
            'latent_r2_std': metrics['std_latent_r2'],
        })
    
    # Sort by configuration key for consistent ordering
    configs = sorted(configs, key=lambda x: (x['l2_raw'], x['l2_latent']))
    
    n_configs = len(configs)
    logger.info(f"Plotting R² for {n_configs} configurations...")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(max(16, n_configs * 0.4), 10))
    
    x_pos = np.arange(n_configs)
    labels = [c['label'] for c in configs]
    
    # Panel 1: Raw R²
    ax = axes[0]
    raw_r2 = [c['raw_r2'] for c in configs]
    raw_r2_std = [c['raw_r2_std'] for c in configs]
    
    bars = ax.bar(x_pos, raw_r2, yerr=raw_r2_std, capsize=3, 
                  alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if raw_r2[i] >= 0.7:
            bar.set_color('green')
        elif raw_r2[i] >= 0.5:
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'Raw Features R² - {dataset_label}\n(Green: >0.7, Blue: 0.5-0.7, Red: <0.5)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add mean line
    mean_raw = np.mean(raw_r2)
    ax.axhline(mean_raw, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_raw:.3f}')
    ax.legend()
    
    # Panel 2: Latent R²
    ax = axes[1]
    latent_r2 = [c['latent_r2'] for c in configs]
    latent_r2_std = [c['latent_r2_std'] for c in configs]
    
    bars = ax.bar(x_pos, latent_r2, yerr=latent_r2_std, capsize=3, 
                  alpha=0.7, color='darkorange', edgecolor='black', linewidth=0.5)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if latent_r2[i] >= 0.9:
            bar.set_color('green')
        elif latent_r2[i] >= 0.7:
            bar.set_color('darkorange')
        else:
            bar.set_color('coral')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'Latent Features R² - {dataset_label}\n(Green: >0.9, Orange: 0.7-0.9, Red: <0.7)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0.7, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add mean line
    mean_latent = np.mean(latent_r2)
    ax.axhline(mean_latent, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_latent:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    # Save with descriptive filename
    save_path = output_dir / f"r2_all_configs_{dataset_label.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved R² plot: {save_path}")
    plt.close()


def plot_r2_comparison(tuning_results: dict, output_dir: Path, dataset_label: str = ""):
    """Plot raw vs latent R² side-by-side comparison."""
    
    output_dir = Path(output_dir)
    
    configs = []
    for config_key, metrics in tuning_results.items():
        configs.append({
            'label': f"R:{metrics['l2_raw']:.0e}\nL:{metrics['l2_latent']:.0e}",
            'l2_raw': metrics['l2_raw'],
            'l2_latent': metrics['l2_latent'],
            'raw_r2': metrics['avg_raw_r2'],
            'latent_r2': metrics['avg_latent_r2'],
            'raw_r2_std': metrics['std_raw_r2'],
            'latent_r2_std': metrics['std_latent_r2'],
        })
    
    configs = sorted(configs, key=lambda x: (x['l2_raw'], x['l2_latent']))
    
    n_configs = len(configs)
    logger.info(f"Plotting R² comparison for {n_configs} configurations...")
    
    fig, ax = plt.subplots(figsize=(max(18, n_configs * 0.5), 8))
    
    x_pos = np.arange(n_configs)
    width = 0.35
    labels = [c['label'] for c in configs]
    
    raw_r2 = [c['raw_r2'] for c in configs]
    latent_r2 = [c['latent_r2'] for c in configs]
    raw_r2_std = [c['raw_r2_std'] for c in configs]
    latent_r2_std = [c['latent_r2_std'] for c in configs]
    
    # Side-by-side bars
    ax.bar(x_pos - width/2, raw_r2, width, yerr=raw_r2_std, capsize=3,
           alpha=0.8, color='steelblue', label='Raw Features', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width/2, latent_r2, width, yerr=latent_r2_std, capsize=3,
           alpha=0.8, color='darkorange', label='Latent Features', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='right')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'Raw vs Latent R² Comparison - {dataset_label}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    save_path = output_dir / f"r2_comparison_{dataset_label.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved comparison plot: {save_path}")
    plt.close()


def extract_rmse_from_results_files(tuning_base_dir: Path, tuning_results: dict):
    """
    Extract RMSE values from individual result JSON files.
    
    Since tuning_results.json doesn't contain RMSE, we need to read from
    the individual configuration result files.
    """
    logger.info("Extracting RMSE from individual result files...")
    
    rmse_data = {}
    
    for config_key, metrics in tuning_results.items():
        # Find the directory for this configuration
        config_dir = tuning_base_dir / config_key
        
        if not config_dir.exists():
            logger.warning(f"Config directory not found: {config_dir}")
            continue
        
        # Look for result JSON files
        result_files = list(config_dir.glob("results_*_cv*.json"))
        
        if not result_files:
            logger.warning(f"No result files found in {config_dir}")
            continue
        
        raw_rmse_list = []
        latent_rmse_list = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Extract RMSE from each train size
                for train_size_key, train_metrics in result_data.items():
                    if isinstance(train_metrics, dict) and 'raw_accuracy' in train_metrics:
                        raw_rmse_list.append(float(train_metrics['raw_accuracy']['rmse']))
                        latent_rmse_list.append(float(train_metrics['latent_accuracy']['rmse']))
            
            except Exception as e:
                logger.warning(f"Error reading {result_file}: {e}")
                continue
        
        if raw_rmse_list:
            rmse_data[config_key] = {
                'raw_rmse_mean': np.mean(raw_rmse_list),
                'raw_rmse_std': np.std(raw_rmse_list),
                'latent_rmse_mean': np.mean(latent_rmse_list),
                'latent_rmse_std': np.std(latent_rmse_list),
            }
    
    logger.info(f"Extracted RMSE for {len(rmse_data)} configurations")
    return rmse_data


def plot_rmse_metrics(tuning_results: dict, rmse_data: dict, output_dir: Path, dataset_label: str = ""):
    """Plot RMSE for all configurations."""
    
    if not rmse_data:
        logger.warning("No RMSE data available. Skipping RMSE plots.")
        return
    
    output_dir = Path(output_dir)
    
    # Extract and sort data
    configs = []
    for config_key, metrics in tuning_results.items():
        if config_key not in rmse_data:
            continue
        
        rmse = rmse_data[config_key]
        configs.append({
            'label': f"R:{metrics['l2_raw']:.0e} L:{metrics['l2_latent']:.0e}",
            'l2_raw': metrics['l2_raw'],
            'l2_latent': metrics['l2_latent'],
            'raw_rmse': rmse['raw_rmse_mean'],
            'latent_rmse': rmse['latent_rmse_mean'],
            'raw_rmse_std': rmse['raw_rmse_std'],
            'latent_rmse_std': rmse['latent_rmse_std'],
        })
    
    configs = sorted(configs, key=lambda x: (x['l2_raw'], x['l2_latent']))
    
    n_configs = len(configs)
    logger.info(f"Plotting RMSE for {n_configs} configurations...")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(max(16, n_configs * 0.4), 10))
    
    x_pos = np.arange(n_configs)
    labels = [c['label'] for c in configs]
    
    # Panel 1: Raw RMSE
    ax = axes[0]
    raw_rmse = [c['raw_rmse'] for c in configs]
    raw_rmse_std = [c['raw_rmse_std'] for c in configs]
    
    bars = ax.bar(x_pos, raw_rmse, yerr=raw_rmse_std, capsize=3, 
                  alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Raw Features RMSE - {dataset_label}\n(Lower is better)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add mean line
    mean_raw = np.mean(raw_rmse)
    ax.axhline(mean_raw, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_raw:.4f}')
    ax.legend()
    
    # Panel 2: Latent RMSE
    ax = axes[1]
    latent_rmse = [c['latent_rmse'] for c in configs]
    latent_rmse_std = [c['latent_rmse_std'] for c in configs]
    
    bars = ax.bar(x_pos, latent_rmse, yerr=latent_rmse_std, capsize=3, 
                  alpha=0.7, color='darkorange', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Latent Features RMSE - {dataset_label}\n(Lower is better)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add mean line
    mean_latent = np.mean(latent_rmse)
    ax.axhline(mean_latent, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_latent:.4f}')
    ax.legend()
    
    plt.tight_layout()
    
    save_path = output_dir / f"rmse_all_configs_{dataset_label.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved RMSE plot: {save_path}")
    plt.close()


def plot_rmse_comparison(tuning_results: dict, rmse_data: dict, output_dir: Path, dataset_label: str = ""):
    """Plot raw vs latent RMSE side-by-side comparison."""
    
    if not rmse_data:
        return
    
    output_dir = Path(output_dir)
    
    configs = []
    for config_key, metrics in tuning_results.items():
        if config_key not in rmse_data:
            continue
        
        rmse = rmse_data[config_key]
        configs.append({
            'label': f"R:{metrics['l2_raw']:.0e}\nL:{metrics['l2_latent']:.0e}",
            'l2_raw': metrics['l2_raw'],
            'l2_latent': metrics['l2_latent'],
            'raw_rmse': rmse['raw_rmse_mean'],
            'latent_rmse': rmse['latent_rmse_mean'],
            'raw_rmse_std': rmse['raw_rmse_std'],
            'latent_rmse_std': rmse['latent_rmse_std'],
        })
    
    configs = sorted(configs, key=lambda x: (x['l2_raw'], x['l2_latent']))
    
    n_configs = len(configs)
    logger.info(f"Plotting RMSE comparison for {n_configs} configurations...")
    
    fig, ax = plt.subplots(figsize=(max(18, n_configs * 0.5), 8))
    
    x_pos = np.arange(n_configs)
    width = 0.35
    labels = [c['label'] for c in configs]
    
    raw_rmse = [c['raw_rmse'] for c in configs]
    latent_rmse = [c['latent_rmse'] for c in configs]
    raw_rmse_std = [c['raw_rmse_std'] for c in configs]
    latent_rmse_std = [c['latent_rmse_std'] for c in configs]
    
    # Side-by-side bars
    ax.bar(x_pos - width/2, raw_rmse, width, yerr=raw_rmse_std, capsize=3,
           alpha=0.8, color='steelblue', label='Raw Features', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width/2, latent_rmse, width, yerr=latent_rmse_std, capsize=3,
           alpha=0.8, color='darkorange', label='Latent Features', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='right')
    ax.set_ylabel('RMSE (Lower is better)', fontsize=12)
    ax.set_title(f'Raw vs Latent RMSE Comparison - {dataset_label}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = output_dir / f"rmse_comparison_{dataset_label.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"✓ Saved RMSE comparison: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot R² and RMSE metrics for all L2 configurations.'
    )
    
    parser.add_argument('--tuning_results', type=str, required=True,
                       help='Path to tuning_results.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--dataset_label', type=str, default='',
                       help='Dataset label for plot titles')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PLOTTING L2 TUNING METRICS")
    logger.info("=" * 80)
    logger.info(f"Input: {args.tuning_results}")
    logger.info(f"Output: {args.output_dir}")
    
    # Load results
    with open(args.tuning_results, 'r') as f:
        tuning_results = json.load(f)
    
    logger.info(f"Loaded {len(tuning_results)} configurations")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot R² metrics
    plot_r2_metrics(tuning_results, output_dir, args.dataset_label)
    plot_r2_comparison(tuning_results, output_dir, args.dataset_label)
    
    # Extract and plot RMSE metrics
    tuning_base_dir = Path(args.tuning_results).parent
    rmse_data = extract_rmse_from_results_files(tuning_base_dir, tuning_results)
    
    if rmse_data:
        plot_rmse_metrics(tuning_results, rmse_data, output_dir, args.dataset_label)
        plot_rmse_comparison(tuning_results, rmse_data, output_dir, args.dataset_label)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PLOTTING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output files in: {output_dir}")
    logger.info(f"  - r2_all_configs_{args.dataset_label.lower().replace(' ', '_')}.png")
    logger.info(f"  - r2_comparison_{args.dataset_label.lower().replace(' ', '_')}.png")
    if rmse_data:
        logger.info(f"  - rmse_all_configs_{args.dataset_label.lower().replace(' ', '_')}.png")
        logger.info(f"  - rmse_comparison_{args.dataset_label.lower().replace(' ', '_')}.png")


if __name__ == "__main__":
    main()

