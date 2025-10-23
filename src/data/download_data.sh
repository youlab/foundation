#!/usr/bin/env bash
set -euo pipefail

echo "=== Foundation Model Data Download Script ==="
echo "This script downloads all datasets from Hugging Face using Git LFS."
echo "Prerequisites: SSH key authentication must be set up (see README)."
echo

# Load git-lfs module if on HPC (will be ignored on other systems)
if command -v module &>/dev/null; then
    echo "Loading git-lfs module (if available)..."
    module load git-lfs 2>/dev/null || echo "Note: git-lfs module not found, assuming system installation"
fi

# Ensure git-lfs is available
if ! command -v git-lfs &>/dev/null; then
    echo "ERROR: git-lfs not found!" >&2
    echo "Please follow the setup instructions in the README:" >&2
    echo "  - HPC: module load git-lfs" >&2
    echo "  - Local: install from https://git-lfs.github.com/" >&2
    exit 1
fi

# Initialize git-lfs (idempotent operation)
echo "Initializing git-lfs..."
git lfs install --local

# Quick authentication check
echo "Checking SSH authentication..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -T git@hf.co 2>&1 | grep -q "Hi "; then
    echo "ERROR: SSH authentication to Hugging Face failed!" >&2
    echo "Please set up SSH keys following the README instructions:" >&2
    echo "  1. Generate SSH key: ssh-keygen -t ed25519 -C \"your_email@example.com\"" >&2
    echo "  2. Add public key to https://huggingface.co/settings/keys" >&2
    echo "  3. Test with: ssh -T git@hf.co" >&2
    echo "You must see 'Hi your_username!' (not 'anonymous')" >&2
    exit 1
fi

echo "✓ Authentication verified"
echo

# Function to download dataset
download_dataset() {
    local repo_name=$1
    local local_name=$2
    
    echo "=== Downloading $local_name ==="
    
    # Remove existing directory
    if [ -d "$local_name" ]; then
        echo "Removing existing $local_name directory..."
        rm -rf "$local_name"
    fi
    
    # Clone repository
    echo "Cloning $repo_name..."
    git clone "git@hf.co:datasets/you-lab/$repo_name" "$local_name"
    
    # Download LFS files
    echo "Downloading large files (this may take several minutes)..."
    cd "$local_name"
    
    # Show progress of LFS download
    git lfs fetch
    git lfs checkout
    
    # Quick verification
    local lfs_files=$(git lfs ls-files | wc -l)
    echo "✓ Downloaded $lfs_files LFS files"
    
    cd ..
    echo "✓ $local_name completed successfully"
    echo
}

# Download all datasets
echo "Starting downloads..."
echo

download_dataset "foundation-model-data-experimental" "experimental"
download_dataset "foundation-model-data-simulation" "simulation" 
download_dataset "foundation-model-data-processed" "processed"

echo "=== All Downloads Complete ==="
echo
echo "Dataset directories created:"
echo "  - experimental/  ($(du -sh experimental 2>/dev/null | cut -f1 || echo "unknown size"))"
echo "  - simulation/    ($(du -sh simulation 2>/dev/null | cut -f1 || echo "unknown size"))"  
echo "  - processed/     ($(du -sh processed 2>/dev/null | cut -f1 || echo "unknown size"))"
echo
echo "All datasets are ready for use!"