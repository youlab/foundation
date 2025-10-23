# foundations
This is a repository for the code for the foundation model for microbial growth.

# Running code
## Machine Learning
Use the following commands to run a script. This is necessary for training models and working with PyTorch.
```
srun -p youlab-gpu --mem=16000 --mincpu=8 --gres=gpu:1 --account=youlab --pty bash -i
apptainer exec --nv /hpc/group/youlab/you-lab-deep-learning.sif python -m <filename>
```
Alternatively, you can work in a shell. This reduces the time to load the container if you are going to be running a lot of scripts.
```
srun -p youlab-gpu --mem=16000 --mincpu=8 --gres=gpu:1 --account=youlab --pty bash -i
apptainer shell --nv /hpc/group/youlab/you-lab-deep-learning.sif
```

### Data Preparation
* `src/data/normalize.py`: this file normalizes the data from the experiments and simulations.
* `src/data/compile.py`: this file compiles the normalized data into one large dataset and then splits into test and train sets.

To pull the container:
`curl -O https://research-singularity-registry.oit.duke.edu/zah8/you-lab-deep-learning.sif`


### Commands for Optuna
To run a study using Optuna, you must first create the study in the relevant database. Ensure the database name here matches the database defined in `ml.config.py`. Optuna allows you to run the optimization on different cores, so this can be run concurrently. Reference the Optuna documentation for more details: [Optuna docs](https://optuna.readthedocs.io/en/stable/index.html).
`optuna create-study --study-name "my-study" --storage "sqlite:///studies.db"`



### Prerequisites for Data Download

**On HPC systems, load the Git LFS module:**
```bash
module load git-lfs/3.6.0  # or latest available version
module avail git           # to see available versions
```

**Initialize Git LFS for your user account (one-time setup):**
```bash
git lfs install
```

### Step 3: Authentication Setup (REQUIRED)

**IMPORTANT:** You MUST set up authentication before running the data download script, or it will fail with authentication errors.

Since Hugging Face requires authentication for dataset access, you need to set up SSH keys:

#### SSH Key Setup (Required for download_data.sh)
1. **Generate SSH key on your system:**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   Press Enter to accept default locations.

2. **Add your public key to Hugging Face:**
   - Copy your public key: `cat ~/.ssh/id_ed25519.pub`
   - Go to https://huggingface.co/settings/keys
   - Paste and save the public key

3. **Test connection (CRITICAL STEP):**
   ```bash
   ssh -T git@hf.co
   ```
   **You MUST see:** `Hi your_username!` 
   
   **If you see:** `Hi anonymous!` - your SSH key is not properly set up. Repeat steps 1-2.

4. **Add key to SSH agent (if needed):**
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

## User-specific updates
* `/foundations/src/config.py`: ensure that `DIR_SRC` is defined correctly based on where you save this repository.
* `/foundations/src/data/simulation/approved_files.sh`: replace the hard-coded paths for moving the approved files over.
* `/foundations/src/slurm_scripts`: replace the hard-coded paths for all of the files and data locations.
* `/foundations/src/data`: use supplied code to download all of the relevant files from HuggingFace. While the files can be downloaded with each use, if you are going to be using this repository extensively it is recommended to download the entire datasets. You can use the `download_data.sh` file in the directory to delete the existing directories and download all of the data from HuggingFace, or you can manually run the following commands: 
    * `git clone git@hf.co:datasets/you-lab/foundation-model-data-experimental`
    * `git clone git@hf.co:datasets/you-lab/foundation-model-data-simulation`
    * `git clone git@hf.co:datasets/you-lab/foundation-model-data-processed`
* `/foundations/app/data`: use supplied code to download all of the relevant files from HuggingFace. While the files can be downloaded with each use, if you are going to be using this repository extensively it is recommended to download the entire datasets. You can use the `download_data.sh` file in the directory to delete the existing directories and download all of the data from HuggingFace, or you can manually run the following commands: 
    * `git clone git@hf.co:datasets/you-lab/foundation-model-results-experimental-consortia`
    * `git clone git@hf.co:datasets/you-lab/foundation-model-results-simulation-consortia`


### Downloading the data 
The datasets use Git LFS (Large File Storage) for large files. 

**Recommended approach:** Use the `download_data.sh` script in `/foundations/src/data`:
```bash
cd src/data
./download_data.sh
```

This script will:
- Check for Git LFS availability
- Remove any existing dataset directories  
- Clone all three datasets from Hugging Face
- Download all large files using Git LFS
- Rename directories to `experimental`, `simulation`, and `processed`

**Manual approach:** If the script fails or you want to download individually:

```bash
cd src/data

# Clone each dataset
git clone git@hf.co:datasets/you-lab/foundation-model-data-experimental.git
git clone git@hf.co:datasets/you-lab/foundation-model-data-simulation.git
git clone git@hf.co:datasets/you-lab/foundation-model-data-processed.git

# Download the large files for each dataset
cd foundation-model-data-experimental
git lfs fetch && git lfs checkout
cd ../foundation-model-data-simulation  
git lfs fetch && git lfs checkout
cd ../foundation-model-data-processed
git lfs fetch && git lfs checkout
cd ..

# Rename directories (optional, to match script behavior)
mv foundation-model-data-experimental experimental
mv foundation-model-data-simulation simulation  
mv foundation-model-data-processed processed
```

### Troubleshooting

- **"Authentication required" errors**: 
  - Make sure you completed Step 3 (SSH key setup)
  - Test with `ssh -T git@hf.co` - you MUST see your username, not "anonymous"
  - If you see "Could not open a connection to your authentication agent", run:
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    ```

- **"git-lfs not found"**: 
  - On HPC: `module load git-lfs`
  - On local systems: install from https://git-lfs.github.com/

- **"Git LFS is not installed for this repository"**: 
  - Run `git lfs install` after loading the module

- **Empty directories after download**: 
  - The script handles this automatically with `git lfs fetch` and `git lfs checkout`
  - If using manual approach, make sure to run these commands in each dataset directory

- **Script stops at authentication**: 
  - Double-check your SSH key is properly added to Hugging Face
  - Verify the key works: `ssh -T git@hf.co`

### Storage Requirements

These datasets contain large files managed by Git LFS. Ensure you have sufficient storage space:
- foundation-model-data-experimental: ~1.9 G
- foundation-model-data-simulation: ~324 M
- foundation-model-data-processed: 1.7 G

While the files can be downloaded with each use, if you are going to be using this repository extensively it is recommended to download the entire datasets locally.

# Antibiotics Pipeline

## Sequential Execution Requirements

**Important:** The models in `run_applications.py` must be run sequentially due to dependencies between components. Follow this order:

### 1. Consortia Simulation Models (Prerequisites)
Before running any other models, you need to generate the regression models:

* **First, run:** `RUN_CONSORTIA_SIM_V2 = True` (Run with SLURM array 0-19)
  * **Purpose:** Creates regression models for microbial consortia simulations
  * **Generates:** `regr_*.pkl` files in `src/applications/consortia/cache/`
  * **Configurations:** Handles both simple and complex datasets with latent and segment128 target types
  * **Cross-validation:** Runs 5-fold cross-validation (cross_val 0-4) for each configuration
  * **Train sizes:** Generates models for multiple training sizes (12800, 51200, 102400)
  * **Required:** You must run this to generate regression files before proceeding

* **Then run:** `RUN_CONSORTIA_SIM_FORECAST_V2 = True` (Run with SLURM array 0-19)
  * **Purpose:** Uses the regression models to generate future predictions/forecasts
  * **Dependency:** Requires the regression files generated from `RUN_CONSORTIA_SIM_V2`
  * **Generates:** `future_outlook_*.npz` files in the cache directory
  * **File naming:** Uses `regr_latent_*` files for latent target types, `regr_raw_*` for segment128 types
  * Will not run successfully without the regression files from `RUN_CONSORTIA_SIM_V2`

### 2. Antibiotic Models
* **Run:** `RUN_ANTIBIOTIC_MAIN = True` 
  * **Purpose:** Trains and evaluates antibiotic resistance prediction models
  * **Cross-validation:** Runs 5-fold cross-validation (0-4)
  * **Datasets:** Processes both Kyeri and Carolyn antibiotic datasets
  * **Required before:** Running antibiotic summaries
  
* **Then run:** `RUN_ANTIBIOTIC_SUMMARIES = True`
  * **Purpose:** Generates summary statistics and analysis from antibiotic model results
  * **Dependency:** Must run `RUN_ANTIBIOTIC_MAIN` first
  * **Outputs:** Creates summary files for downstream analysis and visualization

### 3. Consortia Experimental Models
* **Run:** `RUN_CONSORTIA_EXP = True` (Run with SLURM array 0-479)
  * **Purpose:** Processes experimental consortia data and generates predictions
  * **Array size:** 480 total tasks (0-479)
  * **Cache handling:** Removes existing cache files and retries failed tasks

### 4. Summary and Final Analysis (Run Last)
These require all previous steps to be completed:

* **Run:** `RUN_CONSORTIA_SIM_FOCUSED_SUMMARY = True` (no array needed)
  * **Purpose:** Generates focused analysis and summary of consortia simulation results
  * **Dependency:** Requires all previous consortia simulation files to be completed
  * **Outputs:** Epsilon analysis and main summary statistics

* **Run:** `RUN_CONSORTIA_EXP_FOCUSED_SUMMARY = True` (no array needed)
  * **Purpose:** Generates focused analysis and summary of consortia experimental results
  * **Dependency:** Requires all previous consortia experimental files to be completed
  * **Outputs:** Epsilon analysis for experimental data

* **Run:** `RUN_CONSORTIA_EXP_FINAL_ABUNDANCE_CLASSIFICATION = True` (no array needed)
  * **Purpose:** Performs final abundance classification analysis on experimental consortia data
  * **Parameters:** Uses threshold=10, points_back=1 (configurable in code)
  * **Array options:** Can be configured for different threshold/points_back combinations

* **Optional:** `RUN_SUPER_RESOLUTION = True`
  * **Purpose:** Applies super-resolution techniques to enhance data resolution
  * **Dependency:** Uses task_id from SLURM environment

## Troubleshooting Missing Files

### Quick Reference: Task ID Mappings
The SLURM array tasks (0-19) for `RUN_CONSORTIA_SIM_V2` and `RUN_CONSORTIA_SIM_FORECAST_V2` map to different configurations:
- **Tasks 0-4:** Simple dataset, segment128 target (cross-validation 0-4)
- **Tasks 5-9:** Simple dataset, latent target (cross-validation 0-4) 
- **Tasks 10-14:** Complex dataset, segment128 target (cross-validation 0-4)
- **Tasks 15-19:** Complex dataset, latent target (cross-validation 0-4)

Each task generates files for 3 training sizes: 12800, 51200, and 102400.

### Common Issue: "FileNotFoundError" for Regression Models

**Problem:** You may encounter errors like:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/path/to/regr_latent_simple_latent_A7X_8_3_12800.pkl'
```

**Root Cause:** This happens when `RUN_CONSORTIA_SIM_FORECAST_V2` is enabled but the required regression files from `RUN_CONSORTIA_SIM_V2` are missing or incomplete.

**Solution:**
1. **Identify missing files:** Check the SLURM error logs to see which specific files are missing
2. **Clean incomplete files:** Delete any partial/incomplete regression files to force regeneration:
   ```bash
   # Delete incomplete simple files (example)
   rm /path/to/cache/regr_*simple*A7X_8*.pkl
   rm /path/to/cache/future_outlook_*simple*A7X_8*.npz
   ```
3. **Enable regeneration:** In `run_applications.py`, set:
   ```python
   RUN_CONSORTIA_SIM_V2 = True           # Enable to generate missing files
   RUN_CONSORTIA_SIM_FORECAST_V2 = False  # Disable temporarily
   ```
4. **Run with appropriate array:** Use SLURM array 0-19 to regenerate all configurations
5. **After completion:** Switch back to forecast mode:
   ```python
   RUN_CONSORTIA_SIM_V2 = False          # Disable to avoid regenerating
   RUN_CONSORTIA_SIM_FORECAST_V2 = True   # Re-enable for forecasting
   ```

### Common Issue: JSON Cache Files Preventing Regeneration

**Problem:** You run `RUN_CONSORTIA_SIM_V2` but the script completes immediately with "Results file already exists, returning early" without generating the actual regression `.pkl` files.

**Root Cause:** The `future_v2.py` script checks for JSON result files (e.g., `consortia_regression_9_A7X_8.json`) and exits early if they exist, even if the actual regression `.pkl` files were never created or are incomplete.

**Solution:**
1. **Check for JSON files in cache directory:**
   ```bash
   cd /path/to/foundations/src/applications/consortia/cache/
   ls consortia_regression_*.json
   ```

2. **Delete JSON files for tasks that need regeneration:**
   ```bash
   # For simple dataset tasks (0-9):
   rm consortia_regression_{0..9}_A7X_8.json
   
   # For complex dataset tasks (10-19):
   rm consortia_regression_{10..19}_A7X_8.json
   
   # Or delete all to force complete regeneration:
   rm consortia_regression_*_A7X_8.json
   ```

3. **Re-run SLURM jobs:** After deleting JSON files, re-run the SLURM array for the affected task IDs

**File Patterns to Check:**
- `regr_latent_simple_latent_A7X_8_*` (cross-validation 0-4, all train sizes)
- `regr_raw_simple_latent_A7X_8_*` (cross-validation 0-4, all train sizes)  
- `regr_*_simple_segment128_A7X_8_*` (cross-validation 0-4, all train sizes)
- `regr_*_complex_*_A7X_8_*` (should be complete if only simple files are missing)

**Prevention:** Always run `RUN_CONSORTIA_SIM_V2` completely before enabling `RUN_CONSORTIA_SIM_FORECAST_V2`.

## Alternative: Full Model Comparison
To run all models for comparison, use `run_comparison.py`:
* Set `RUN_ANTIBIOTICS_FULL = True`, then run using SLURM with an array of 41
* After completion, run `run_comparison.py` with `RUN_ANTIBIOTICS_SUMMARY = True`

## Figure Generation (`run_figs.py`)

### Data-Only Figures (No application required)
These rely solely on existing data and can be run independently:
* `RUN_ANTIBIOTICS_KYERI_TOP10F_FIG_S4`
* `RUN_ANTIBIOTICS_KYERI_KEIO_FIG_S5`
* `RUN_ANTIBIOTICS_KYERI_TIMER_FIG_S6`
* `RUN_ANTIBIOTICS_CAROLYN_ALL_FIG_S7`

### Primary Model Figures
* `RUN_ANTIBIOTICS_FIG_3` (requires primary model only)

### Model Comparison Figures
* `RUN_ANTIBIOTICS_MODEL_COMPARISON_FIG_S8` (requires all models to be completed)
