#!/bin/bash

# Set target extraction directory
TARGET_DIR="/scratch/project_462000938/vidor/video"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Clone the dataset repo with LFS enabled
echo "Cloning Vidor dataset..."
git lfs install
git clone https://huggingface.co/datasets/shangxd/vidor

cd vidor || { echo "Failed to enter vidor directory"; exit 1; }

echo "Downloading files with Git LFS..."
git lfs pull

echo "Moving zip files to parent directory..."
find . -name "*.zip" -exec mv {} ../ \;

cd ..
rm -rf vidor

echo "Unzipping dataset to $TARGET_DIR ..."
for f in *.zip; do
  echo "Extracting $f ..."
  unzip -q "$f" -d "$TARGET_DIR"
done

# Remove zip files after extraction
for f in *.zip; do
  echo "Removing $f ..."
  rm "$f"
done

echo "All done!"
