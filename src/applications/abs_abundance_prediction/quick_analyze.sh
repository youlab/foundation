#!/bin/bash
# Quick analysis script for L2 regularization tuning results

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_BASE="/hpc/dctrl/is178/foundation/results/abs_abundance_prediction"

echo "=========================================="
echo "L2 Regularization Tuning Analysis"
echo "=========================================="

# Check if tuning results exist
GLV_RESULTS="${RESULTS_BASE}/glv_regularization_tuning/tuning_results.json"
CHAOTIC_RESULTS="${RESULTS_BASE}/chaotic_regularization_tuning/tuning_results.json"

if [ -f "$GLV_RESULTS" ] && [ -f "$CHAOTIC_RESULTS" ]; then
    echo "✓ Found tuning results for both datasets"
    echo "  GLV: $GLV_RESULTS"
    echo "  Chaotic: $CHAOTIC_RESULTS"
    echo ""
    echo "Running comparative analysis..."
    
    OUTPUT_DIR="${RESULTS_BASE}/comparative_analysis"
    
    python "${SCRIPT_DIR}/analyze_tuning_results.py" \
        --glv_results "$GLV_RESULTS" \
        --chaotic_results "$CHAOTIC_RESULTS" \
        --output_dir "$OUTPUT_DIR"
    
    echo ""
    echo "=========================================="
    echo "✓ Analysis complete!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - glv_analysis/"
    echo "    - comprehensive_r2_analysis.png"
    echo "    - l2_effect_analysis.png"
    echo "    - statistical_summary.png"
    echo "  - chaotic_analysis/"
    echo "    - comprehensive_r2_analysis.png"
    echo "    - l2_effect_analysis.png"
    echo "    - statistical_summary.png"
    echo "  - dataset_comparison.png"
    
elif [ -f "$GLV_RESULTS" ]; then
    echo "✓ Found GLV tuning results"
    echo "  Path: $GLV_RESULTS"
    echo ""
    echo "Running GLV analysis..."
    
    OUTPUT_DIR="${RESULTS_BASE}/glv_regularization_tuning/analysis"
    
    python "${SCRIPT_DIR}/analyze_tuning_results.py" \
        --tuning_results "$GLV_RESULTS" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_label "GLV"
    
    echo ""
    echo "✓ GLV analysis complete!"
    echo "Results saved to: $OUTPUT_DIR"
    
elif [ -f "$CHAOTIC_RESULTS" ]; then
    echo "✓ Found Chaotic tuning results"
    echo "  Path: $CHAOTIC_RESULTS"
    echo ""
    echo "Running Chaotic analysis..."
    
    OUTPUT_DIR="${RESULTS_BASE}/chaotic_regularization_tuning/analysis"
    
    python "${SCRIPT_DIR}/analyze_tuning_results.py" \
        --tuning_results "$CHAOTIC_RESULTS" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_label "Chaotic"
    
    echo ""
    echo "✓ Chaotic analysis complete!"
    echo "Results saved to: $OUTPUT_DIR"
    
else
    echo "✗ No tuning results found!"
    echo ""
    echo "Expected locations:"
    echo "  - $GLV_RESULTS"
    echo "  - $CHAOTIC_RESULTS"
    echo ""
    echo "Please run the tuning script first:"
    echo "  sbatch src/slurm_scripts/run_app.sh"
    exit 1
fi

