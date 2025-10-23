#!/usr/bin/env bash
set -euo pipefail

echo "=== Foundation Model Consortia Results Download Script ==="
echo "This script downloads consortia results from Hugging Face."
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

# Function to download consortia results
download_consortia() {
    local repo_name=$1
    local local_dir=$2
    local cache_dir="$local_dir/cache"
    
    echo "=== Downloading $local_dir ==="
    
    # Remove existing cache directory
    if [ -d "$cache_dir" ]; then
        echo "Removing existing $cache_dir directory..."
        rm -rf "$cache_dir"
    fi
    
    # Create parent directory
    echo "Creating $local_dir directory..."
    mkdir -p "$local_dir"
    
    # Clone repository
    echo "Cloning $repo_name..."
    git clone "git@hf.co:datasets/you-lab/$repo_name"
    
    # Move to cache location
    echo "Moving to $cache_dir..."
    mv "$repo_name" "$cache_dir"
    
    # Download LFS files if any
    echo "Downloading large files (if any)..."
    cd "$cache_dir"
    
    # Check if there are LFS files and download them
    if git lfs ls-files | grep -q .; then
        git lfs fetch
        git lfs checkout
        local lfs_files=$(git lfs ls-files | wc -l)
        echo "✓ Downloaded $lfs_files LFS files"
    else
        echo "✓ No LFS files found"
    fi
    
    cd ../..
    echo "✓ $local_dir completed successfully"
    echo
}

# Download consortia results
echo "Starting downloads..."
echo

download_consortia "foundation-model-results-experimental-consortia" "consortia_exp"
download_consortia "foundation-model-results-simulation-consortia" "consortia"

echo "=== All Downloads Complete ==="
echo
echo "Consortia directories created:"
echo "  - consortia_exp/cache/  ($(du -sh consortia_exp/cache 2>/dev/null | cut -f1 || echo "unknown size"))"
echo "  - consortia/cache/      ($(du -sh consortia/cache 2>/dev/null | cut -f1 || echo "unknown size"))"
echo
echo "All consortia results are ready for use!"