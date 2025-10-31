import logging
from pathlib import Path
import json
import numpy as np
import argparse
import sys

# Import the diagnostic function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from applications.abs_abundance_prediction.abs_abundance_prediction import compare_datasets

logger = logging.getLogger("dataset_comparison")
logger.handlers.clear()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)


def main():
    """Compare two datasets (GLV vs GLV+dispersal) using Diagnostic 4."""
    parser = argparse.ArgumentParser(
        description='Compare GLV vs GLV+Dispersal datasets using Diagnostic 4'
    )
    
    parser.add_argument('--glv_results_dir', type=str, required=True,
                        help='Path to GLV results directory (contains result*.json files)')
    parser.add_argument('--glv_dispersal_results_dir', type=str, required=True,
                        help='Path to GLV+Dispersal results directory (contains result*.json files)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    glv_dir = Path(args.glv_results_dir)
    dispersal_dir = Path(args.glv_dispersal_results_dir)
    
    if not glv_dir.exists():
        logger.error(f"GLV results directory not found: {glv_dir}")
        return
    if not dispersal_dir.exists():
        logger.error(f"GLV+Dispersal results directory not found: {dispersal_dir}")
        return
    
    # Load results from JSON files
    logger.info("Loading GLV results...")
    glv_results = []
    for json_file in sorted(glv_dir.glob("results_glv_mlp_cv*.json")):
        with open(json_file, 'r') as f:
            glv_results.append(json.load(f))
    logger.info(f"Loaded {len(glv_results)} GLV CV folds")
    
    logger.info("Loading GLV+Dispersal results...")
    dispersal_results = []
    for json_file in sorted(dispersal_dir.glob("results_*_mlp_cv*.json")):
        with open(json_file, 'r') as f:
            dispersal_results.append(json.load(f))
    logger.info(f"Loaded {len(dispersal_results)} GLV+Dispersal CV folds")
    
    if not glv_results or not dispersal_results:
        logger.error("No results found in provided directories")
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = glv_dir.parent / "dataset_comparison"
    
    # Run comparison
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC 4: Dataset Comparison (GLV vs GLV+Dispersal)")
    logger.info("=" * 70)
    
    results = [glv_results, dispersal_results]
    labels = ["GLV", "GLV+Dispersal"]
    
    compare_datasets(results, labels, output_dir=args.output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Dataset comparison complete!")
    logger.info(f"Visualization saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
