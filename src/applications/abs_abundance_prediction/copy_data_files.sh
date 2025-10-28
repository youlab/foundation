#!/bin/bash
# Script to copy data files from remote location to local data_files folder
# This makes the paths simpler and keeps data organized

# Create data_files directory if it doesn't exist
mkdir -p data_files

# Source paths (original complicated paths)
SOURCE_RELATIVE="/hpc/group/youlab/is178/absolute_abundance/processed/relative_abundance.npz"
SOURCE_TOTAL="/hpc/group/youlab/is178/absolute_abundance/processed/total_abundance.npz"
SOURCE_CHAOTIC_RELATIVE="/hpc/group/youlab/is178/absolute_abundance/processed/chaotic/relative_abundance.npz"
SOURCE_CHAOTIC_TOTAL="/hpc/group/youlab/is178/absolute_abundance/processed/chaotic/total_abundance.npz"

# Destination paths (simpler local paths)
DEST_RELATIVE="data_files/relative_abundance.npz"
DEST_TOTAL="data_files/total_abundance.npz"
DEST_CHAOTIC_RELATIVE="data_files/chaotic_relative_abundance.npz"
DEST_CHAOTIC_TOTAL="data_files/chaotic_total_abundance.npz"

echo "Copying data files to local data_files folder..."
echo ""

# Copy relative abundance dataset
if [ -f "$SOURCE_RELATIVE" ]; then
    echo "Copying relative abundance dataset..."
    cp "$SOURCE_RELATIVE" "$DEST_RELATIVE"
    echo "  ✓ Copied: $DEST_RELATIVE"
else
    echo "  ❌ ERROR: Source file not found: $SOURCE_RELATIVE"
fi

# Copy total abundance dataset
if [ -f "$SOURCE_TOTAL" ]; then
    echo "Copying total abundance dataset..."
    cp "$SOURCE_TOTAL" "$DEST_TOTAL"
    echo "  ✓ Copied: $DEST_TOTAL"
else
    echo "  ❌ ERROR: Source file not found: $SOURCE_TOTAL"
fi

# Copy chaotic relative abundance dataset
if [ -f "$SOURCE_CHAOTIC_RELATIVE" ]; then
    echo "Copying chaotic relative abundance dataset..."
    cp "$SOURCE_CHAOTIC_RELATIVE" "$DEST_CHAOTIC_RELATIVE"
    echo "  ✓ Copied: $DEST_CHAOTIC_RELATIVE"
else
    echo "  ❌ ERROR: Source file not found: $SOURCE_CHAOTIC_RELATIVE"
fi

# Copy chaotic total abundance dataset
if [ -f "$SOURCE_CHAOTIC_TOTAL" ]; then
    echo "Copying chaotic total abundance dataset..."
    cp "$SOURCE_CHAOTIC_TOTAL" "$DEST_CHAOTIC_TOTAL"
    echo "  ✓ Copied: $DEST_CHAOTIC_TOTAL"
else
    echo "  ❌ ERROR: Source file not found: $SOURCE_CHAOTIC_TOTAL"
fi

echo ""
echo "Done! Data files are now in:"
echo "  - $DEST_RELATIVE"
echo "  - $DEST_TOTAL"
echo "  - $DEST_CHAOTIC_RELATIVE"
echo "  - $DEST_CHAOTIC_TOTAL"