#!/bin/bash
# Simple script to plot R² and RMSE for all L2 configurations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_BASE="/hpc/dctrl/is178/foundation/results/abs_abundance_prediction"

echo "=========================================="
echo "L2 Configuration Metrics Plotting"
echo "=========================================="

# Check for GLV results
GLV_TUNING="${RESULTS_BASE}/glv_regularization_tuning/tuning_results.json"
if [ -f "$GLV_TUNING" ]; then
    echo ""
    echo "✓ Found GLV tuning results"
    echo "  Plotting metrics..."
    
    python "${SCRIPT_DIR}/plot_tuning_metrics.py" \
        --tuning_results "$GLV_TUNING" \
        --output_dir "${RESULTS_BASE}/glv_regularization_tuning/metric_plots" \
        --dataset_label "GLV"
    
    echo "  ✓ GLV plots saved to: ${RESULTS_BASE}/glv_regularization_tuning/metric_plots/"
fi

# Check for Chaotic results
CHAOTIC_TUNING="${RESULTS_BASE}/chaotic_regularization_tuning/tuning_results.json"
if [ -f "$CHAOTIC_TUNING" ]; then
    echo ""
    echo "✓ Found Chaotic tuning results"
    echo "  Plotting metrics..."
    
    python "${SCRIPT_DIR}/plot_tuning_metrics.py" \
        --tuning_results "$CHAOTIC_TUNING" \
        --output_dir "${RESULTS_BASE}/chaotic_regularization_tuning/metric_plots" \
        --dataset_label "Chaotic"
    
    echo "  ✓ Chaotic plots saved to: ${RESULTS_BASE}/chaotic_regularization_tuning/metric_plots/"
fi

# Check if any results were found
if [ ! -f "$GLV_TUNING" ] && [ ! -f "$CHAOTIC_TUNING" ]; then
    echo ""
    echo "✗ No tuning results found!"
    echo ""
    echo "Expected locations:"
    echo "  - $GLV_TUNING"
    echo "  - $CHAOTIC_TUNING"
    echo ""
    echo "Please run the tuning script first:"
    echo "  sbatch src/slurm_scripts/run_app.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Plotting complete!"
echo "=========================================="
echo ""
echo "Generated plots:"
echo "  - r2_all_configs_<dataset>.png       (R² for all configs)"
echo "  - r2_comparison_<dataset>.png        (Raw vs Latent R²)"
echo "  - rmse_all_configs_<dataset>.png     (RMSE for all configs)"
echo "  - rmse_comparison_<dataset>.png      (Raw vs Latent RMSE)"

