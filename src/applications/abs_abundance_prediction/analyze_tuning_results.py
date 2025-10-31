"""
Analyze and visualize L2 regularization tuning results.

This script reads tuning_results.json and creates comprehensive visualizations
showing R² and RMSE performance averaged across all CV folds.

Usage:
    python analyze_tuning_results.py \
        --tuning_results results/chaotic_regularization_tuning/tuning_results.json \
        --output_dir results/chaotic_regularization_tuning/analysis
    
    # Or for both datasets at once:
    python analyze_tuning_results.py \
        --glv_results results/glv_regularization_tuning/tuning_results.json \
        --chaotic_results results/chaotic_regularization_tuning/tuning_results.json \
        --output_dir results/tuning_comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple


def setup_logger():
    """Initialize and return a logger."""
    logger = logging.getLogger("tuning_analyzer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def load_tuning_results(json_path: str) -> Dict:
    """Load tuning results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Tuple[np.ndarray, ...]:
    """
    Extract metrics from tuning results.
    
    Returns:
        Tuple of (l2_raw_values, l2_latent_values, raw_r2, latent_r2, 
                  raw_r2_std, latent_r2_std, raw_r2_trend, latent_advantage)
    """
    configs = []
    for config_key, metrics in results.items():
        configs.append({
            'key': config_key,
            'l2_raw': metrics['l2_raw'],
            'l2_latent': metrics['l2_latent'],
            'avg_raw_r2': metrics['avg_raw_r2'],
            'avg_latent_r2': metrics['avg_latent_r2'],
            'std_raw_r2': metrics['std_raw_r2'],
            'std_latent_r2': metrics['std_latent_r2'],
            'raw_r2_trend': metrics['raw_r2_trend'],
            'latent_advantage': metrics['latent_advantage']
        })
    
    # Convert to arrays
    l2_raw = np.array([c['l2_raw'] for c in configs])
    l2_latent = np.array([c['l2_latent'] for c in configs])
    raw_r2 = np.array([c['avg_raw_r2'] for c in configs])
    latent_r2 = np.array([c['avg_latent_r2'] for c in configs])
    raw_r2_std = np.array([c['std_raw_r2'] for c in configs])
    latent_r2_std = np.array([c['std_latent_r2'] for c in configs])
    raw_r2_trend = np.array([c['raw_r2_trend'] for c in configs])
    latent_advantage = np.array([c['latent_advantage'] for c in configs])
    
    return (l2_raw, l2_latent, raw_r2, latent_r2, 
            raw_r2_std, latent_r2_std, raw_r2_trend, latent_advantage, configs)


def plot_comprehensive_analysis(results: Dict, output_dir: Path, dataset_label: str = ""):
    """Create comprehensive analysis plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Analyzing {len(results)} configurations...")
    
    (l2_raw, l2_latent, raw_r2, latent_r2, 
     raw_r2_std, latent_r2_std, raw_r2_trend, latent_advantage, configs) = extract_metrics(results)
    
    # ========================================================================
    # Figure 1: R² Performance Comparison
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Raw R² vs configurations (sorted)
    ax = axes[0, 0]
    sorted_idx = np.argsort(raw_r2)[::-1]
    x_pos = np.arange(len(raw_r2))
    ax.bar(x_pos, raw_r2[sorted_idx], yerr=raw_r2_std[sorted_idx], 
           alpha=0.7, color='steelblue', capsize=3)
    ax.set_xlabel('Configuration (sorted by performance)')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Raw Features R² (Sorted)\n{dataset_label}')
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(np.mean(raw_r2), color='red', linestyle='--', 
               label=f'Mean: {np.mean(raw_r2):.3f}')
    ax.legend()
    
    # Panel 2: Latent R² vs configurations (sorted)
    ax = axes[0, 1]
    sorted_idx_latent = np.argsort(latent_r2)[::-1]
    ax.bar(x_pos, latent_r2[sorted_idx_latent], yerr=latent_r2_std[sorted_idx_latent], 
           alpha=0.7, color='darkorange', capsize=3)
    ax.set_xlabel('Configuration (sorted by performance)')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Latent Features R² (Sorted)\n{dataset_label}')
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(np.mean(latent_r2), color='red', linestyle='--', 
               label=f'Mean: {np.mean(latent_r2):.3f}')
    ax.legend()
    
    # Panel 3: R² comparison scatter
    ax = axes[1, 0]
    scatter = ax.scatter(raw_r2, latent_r2, c=latent_advantage, 
                        cmap='RdBu_r', s=100, alpha=0.6, edgecolors='k')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal performance')
    ax.set_xlabel('Raw Features R²')
    ax.set_ylabel('Latent Features R²')
    ax.set_title(f'Raw vs Latent R² Comparison\n{dataset_label}')
    ax.grid(alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Latent Advantage')
    
    # Panel 4: Error bars comparison (top 10)
    ax = axes[1, 1]
    top_k = min(10, len(configs))
    sorted_configs = sorted(configs, key=lambda x: x['avg_latent_r2'], reverse=True)[:top_k]
    config_labels = [f"R:{c['l2_raw']:.0e}\nL:{c['l2_latent']:.0e}" 
                     for c in sorted_configs]
    raw_means = [c['avg_raw_r2'] for c in sorted_configs]
    latent_means = [c['avg_latent_r2'] for c in sorted_configs]
    raw_stds = [c['std_raw_r2'] for c in sorted_configs]
    latent_stds = [c['std_latent_r2'] for c in sorted_configs]
    
    x_pos = np.arange(len(config_labels))
    width = 0.35
    ax.barh(x_pos - width/2, raw_means, width, xerr=raw_stds, 
            alpha=0.8, color='steelblue', label='Raw', capsize=3)
    ax.barh(x_pos + width/2, latent_means, width, xerr=latent_stds, 
            alpha=0.8, color='darkorange', label='Latent', capsize=3)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(config_labels, fontsize=8)
    ax.set_xlabel('R² Score')
    ax.set_title(f'Top {top_k} Configurations (by Latent R²)\n{dataset_label}')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = output_dir / "comprehensive_r2_analysis.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved R² analysis to: {save_path}")
    plt.close()
    
    # ========================================================================
    # Figure 2: L2 Effect Analysis
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Raw R² vs L2_raw (for each L2_latent)
    ax = axes[0, 0]
    unique_l2_latent = sorted(set(l2_latent))
    for l2_lat in unique_l2_latent:
        mask = l2_latent == l2_lat
        ax.plot(l2_raw[mask], raw_r2[mask], 'o-', 
                label=f'L2_latent={l2_lat:.0e}', alpha=0.7, markersize=6)
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_xlabel('L2 Regularization (Raw Features)')
    ax.set_ylabel('Raw R² Score')
    ax.set_title(f'Effect of L2_raw on Raw Performance\n{dataset_label}')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel 2: Latent R² vs L2_latent (for each L2_raw)
    ax = axes[0, 1]
    unique_l2_raw = sorted(set(l2_raw))
    for l2_r in unique_l2_raw:
        mask = l2_raw == l2_r
        ax.plot(l2_latent[mask], latent_r2[mask], 's-', 
                label=f'L2_raw={l2_r:.0e}', alpha=0.7, markersize=6)
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_xlabel('L2 Regularization (Latent Features)')
    ax.set_ylabel('Latent R² Score')
    ax.set_title(f'Effect of L2_latent on Latent Performance\n{dataset_label}')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel 3: Raw R² trend analysis
    ax = axes[1, 0]
    scatter = ax.scatter(l2_raw, raw_r2_trend, c=raw_r2, 
                        cmap='RdYlGn', s=100, alpha=0.6, edgecolors='k')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_xlabel('L2 Regularization (Raw Features)')
    ax.set_ylabel('Raw R² Trend\n(negative = declining with more data)')
    ax.set_title(f'Raw R² Trend vs L2_raw\n{dataset_label}')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Raw R²')
    
    # Panel 4: Latent advantage analysis
    ax = axes[1, 1]
    scatter = ax.scatter(l2_raw, l2_latent, c=latent_advantage, 
                        cmap='RdBu_r', s=150, alpha=0.6, edgecolors='k')
    ax.axhline(np.median(l2_latent), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(np.median(l2_raw), color='gray', linestyle='--', alpha=0.3)
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xlabel('L2 Regularization (Raw Features)')
    ax.set_ylabel('L2 Regularization (Latent Features)')
    ax.set_title(f'Latent Advantage Landscape\n{dataset_label}')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Latent R² - Raw R²\n(positive=latent better)')
    
    plt.tight_layout()
    save_path = output_dir / "l2_effect_analysis.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved L2 effect analysis to: {save_path}")
    plt.close()
    
    # ========================================================================
    # Figure 3: Statistical Summary
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Distribution of R² scores
    ax = axes[0, 0]
    ax.hist(raw_r2, bins=20, alpha=0.6, label='Raw', color='steelblue', edgecolor='k')
    ax.hist(latent_r2, bins=20, alpha=0.6, label='Latent', color='darkorange', edgecolor='k')
    ax.axvline(np.mean(raw_r2), color='blue', linestyle='--', 
               label=f'Raw mean: {np.mean(raw_r2):.3f}')
    ax.axvline(np.mean(latent_r2), color='orangered', linestyle='--', 
               label=f'Latent mean: {np.mean(latent_r2):.3f}')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of R² Scores\n{dataset_label}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 2: Uncertainty (std) distribution
    ax = axes[0, 1]
    ax.hist(raw_r2_std, bins=20, alpha=0.6, label='Raw', color='steelblue', edgecolor='k')
    ax.hist(latent_r2_std, bins=20, alpha=0.6, label='Latent', color='darkorange', edgecolor='k')
    ax.axvline(np.mean(raw_r2_std), color='blue', linestyle='--', 
               label=f'Raw mean std: {np.mean(raw_r2_std):.3f}')
    ax.axvline(np.mean(latent_r2_std), color='orangered', linestyle='--', 
               label=f'Latent mean std: {np.mean(latent_r2_std):.3f}')
    ax.set_xlabel('R² Standard Deviation')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cross-Validation Uncertainty\n{dataset_label}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 3: Coefficient of Variation (CV)
    ax = axes[1, 0]
    raw_cv = raw_r2_std / (raw_r2 + 1e-10)
    latent_cv = latent_r2_std / (latent_r2 + 1e-10)
    ax.scatter(raw_r2, raw_cv, alpha=0.6, label='Raw', s=80, edgecolors='k')
    ax.scatter(latent_r2, latent_cv, alpha=0.6, label='Latent', s=80, edgecolors='k')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Coefficient of Variation (std/mean)')
    ax.set_title(f'Stability vs Performance\n{dataset_label}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    STATISTICAL SUMMARY - {dataset_label}
    {'='*50}
    
    RAW FEATURES:
      Mean R²:         {np.mean(raw_r2):.4f} ± {np.std(raw_r2):.4f}
      Median R²:       {np.median(raw_r2):.4f}
      Best R²:         {np.max(raw_r2):.4f}
      Worst R²:        {np.min(raw_r2):.4f}
      Mean Std:        {np.mean(raw_r2_std):.4f}
      Mean Trend:      {np.mean(raw_r2_trend):+.6f}
      % Declining:     {100 * np.mean(raw_r2_trend < -0.001):.1f}%
    
    LATENT FEATURES:
      Mean R²:         {np.mean(latent_r2):.4f} ± {np.std(latent_r2):.4f}
      Median R²:       {np.median(latent_r2):.4f}
      Best R²:         {np.max(latent_r2):.4f}
      Worst R²:        {np.min(latent_r2):.4f}
      Mean Std:        {np.mean(latent_r2_std):.4f}
    
    COMPARISON:
      Latent Advantage:     {np.mean(latent_advantage):+.4f} ± {np.std(latent_advantage):.4f}
      % Latent Better:      {100 * np.mean(latent_advantage > 0):.1f}%
      Max Latent Advantage: {np.max(latent_advantage):+.4f}
      Min Latent Advantage: {np.min(latent_advantage):+.4f}
    
    RECOMMENDATIONS:
      Best overall config (by latent R²):
        L2_raw:    {sorted_configs[0]['l2_raw']:.0e}
        L2_latent: {sorted_configs[0]['l2_latent']:.0e}
        Raw R²:    {sorted_configs[0]['avg_raw_r2']:.4f}
        Latent R²: {sorted_configs[0]['avg_latent_r2']:.4f}
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    save_path = output_dir / "statistical_summary.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved statistical summary to: {save_path}")
    plt.close()
    
    # ========================================================================
    # Print summary to console
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"TUNING RESULTS SUMMARY - {dataset_label}")
    logger.info("=" * 80)
    logger.info(f"\nTotal configurations tested: {len(results)}")
    logger.info(f"\nRAW FEATURES:")
    logger.info(f"  Mean R²: {np.mean(raw_r2):.4f} ± {np.std(raw_r2):.4f}")
    logger.info(f"  Best R²: {np.max(raw_r2):.4f}")
    logger.info(f"  Configurations with declining R²: {100 * np.mean(raw_r2_trend < -0.001):.1f}%")
    logger.info(f"\nLATENT FEATURES:")
    logger.info(f"  Mean R²: {np.mean(latent_r2):.4f} ± {np.std(latent_r2):.4f}")
    logger.info(f"  Best R²: {np.max(latent_r2):.4f}")
    logger.info(f"\nCOMPARISON:")
    logger.info(f"  Average latent advantage: {np.mean(latent_advantage):+.4f}")
    logger.info(f"  Latent better in: {100 * np.mean(latent_advantage > 0):.1f}% of configs")
    logger.info(f"\nTOP 5 CONFIGURATIONS:")
    for i, config in enumerate(sorted_configs[:5], 1):
        logger.info(f"\n{i}. L2_raw={config['l2_raw']:.0e}, L2_latent={config['l2_latent']:.0e}")
        logger.info(f"   Raw R²:    {config['avg_raw_r2']:.4f} ± {config['std_raw_r2']:.4f}")
        logger.info(f"   Latent R²: {config['avg_latent_r2']:.4f} ± {config['std_latent_r2']:.4f}")
        logger.info(f"   Trend:     {config['raw_r2_trend']:+.6f}")


def compare_datasets(glv_path: str, chaotic_path: str, output_dir: Path):
    """Compare tuning results between GLV and Chaotic datasets."""
    
    logger.info("Loading results for both datasets...")
    glv_results = load_tuning_results(glv_path)
    chaotic_results = load_tuning_results(chaotic_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract metrics for both
    glv_metrics = extract_metrics(glv_results)
    chaotic_metrics = extract_metrics(chaotic_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: R² comparison
    ax = axes[0, 0]
    ax.scatter(glv_metrics[2], chaotic_metrics[2], alpha=0.6, s=100, 
              label='Raw', edgecolors='k')
    ax.scatter(glv_metrics[3], chaotic_metrics[3], alpha=0.6, s=100, 
              label='Latent', edgecolors='k')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('GLV R² Score')
    ax.set_ylabel('Chaotic R² Score')
    ax.set_title('Dataset Comparison: R² Scores')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Best configs comparison
    ax = axes[0, 1]
    glv_configs = glv_metrics[8]
    chaotic_configs = chaotic_metrics[8]
    glv_top = sorted(glv_configs, key=lambda x: x['avg_latent_r2'], reverse=True)[:10]
    chaotic_top = sorted(chaotic_configs, key=lambda x: x['avg_latent_r2'], reverse=True)[:10]
    
    config_names = [f"R:{c['l2_raw']:.0e}\nL:{c['l2_latent']:.0e}" for c in glv_top]
    glv_scores = [c['avg_latent_r2'] for c in glv_top]
    chaotic_scores = [c['avg_latent_r2'] for c in chaotic_top]
    
    x_pos = np.arange(len(config_names))
    width = 0.35
    ax.barh(x_pos - width/2, glv_scores, width, alpha=0.8, label='GLV')
    ax.barh(x_pos + width/2, chaotic_scores, width, alpha=0.8, label='Chaotic')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(config_names, fontsize=7)
    ax.set_xlabel('Latent R² Score')
    ax.set_title('Top 10 Configs: GLV vs Chaotic')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    # Panel 3: Latent advantage comparison
    ax = axes[1, 0]
    ax.scatter(glv_metrics[7], chaotic_metrics[7], alpha=0.6, s=100, edgecolors='k')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('GLV Latent Advantage')
    ax.set_ylabel('Chaotic Latent Advantage')
    ax.set_title('Latent Advantage Comparison')
    ax.grid(alpha=0.3)
    
    # Panel 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = f"""
    DATASET COMPARISON
    {'='*50}
    
    GLV DATASET:
      Mean Raw R²:      {np.mean(glv_metrics[2]):.4f}
      Mean Latent R²:   {np.mean(glv_metrics[3]):.4f}
      Best Latent R²:   {np.max(glv_metrics[3]):.4f}
      Mean Advantage:   {np.mean(glv_metrics[7]):+.4f}
    
    CHAOTIC DATASET:
      Mean Raw R²:      {np.mean(chaotic_metrics[2]):.4f}
      Mean Latent R²:   {np.mean(chaotic_metrics[3]):.4f}
      Best Latent R²:   {np.max(chaotic_metrics[3]):.4f}
      Mean Advantage:   {np.mean(chaotic_metrics[7]):+.4f}
    
    DIFFERENCES:
      ΔRaw R²:          {np.mean(chaotic_metrics[2]) - np.mean(glv_metrics[2]):+.4f}
      ΔLatent R²:       {np.mean(chaotic_metrics[3]) - np.mean(glv_metrics[3]):+.4f}
      ΔAdvantage:       {np.mean(chaotic_metrics[7]) - np.mean(glv_metrics[7]):+.4f}
    
    BEST CONFIGS:
      GLV Best:
        L2_raw:    {glv_top[0]['l2_raw']:.0e}
        L2_latent: {glv_top[0]['l2_latent']:.0e}
        Latent R²: {glv_top[0]['avg_latent_r2']:.4f}
      
      Chaotic Best:
        L2_raw:    {chaotic_top[0]['l2_raw']:.0e}
        L2_latent: {chaotic_top[0]['l2_latent']:.0e}
        Latent R²: {chaotic_top[0]['avg_latent_r2']:.4f}
    """
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    save_path = output_dir / "dataset_comparison.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    logger.info(f"Saved dataset comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze L2 regularization tuning results.'
    )
    
    parser.add_argument('--tuning_results', type=str, default=None,
                       help='Path to tuning_results.json file')
    parser.add_argument('--glv_results', type=str, default=None,
                       help='Path to GLV tuning_results.json (for comparison)')
    parser.add_argument('--chaotic_results', type=str, default=None,
                       help='Path to Chaotic tuning_results.json (for comparison)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis plots')
    parser.add_argument('--dataset_label', type=str, default='',
                       help='Dataset label for single dataset analysis')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("TUNING RESULTS ANALYSIS")
    logger.info("=" * 80)
    
    # Single dataset analysis
    if args.tuning_results:
        logger.info(f"Analyzing single dataset: {args.tuning_results}")
        results = load_tuning_results(args.tuning_results)
        plot_comprehensive_analysis(results, args.output_dir, args.dataset_label)
    
    # Cross-dataset comparison
    if args.glv_results and args.chaotic_results:
        logger.info("Comparing GLV and Chaotic datasets...")
        
        # Individual analyses
        glv_results = load_tuning_results(args.glv_results)
        chaotic_results = load_tuning_results(args.chaotic_results)
        
        glv_output = Path(args.output_dir) / "glv_analysis"
        chaotic_output = Path(args.output_dir) / "chaotic_analysis"
        
        plot_comprehensive_analysis(glv_results, glv_output, "GLV")
        plot_comprehensive_analysis(chaotic_results, chaotic_output, "Chaotic")
        
        # Comparison
        compare_datasets(args.glv_results, args.chaotic_results, args.output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

