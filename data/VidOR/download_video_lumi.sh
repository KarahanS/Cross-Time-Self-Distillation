#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────── 1. paths & URLs ──────────────────────────
TARGET_DIR="/scratch/project_462000938/vidor/dataset"   # everything lives here
BASE_URL="https://huggingface.co/datasets/shangxd/vidor/resolve/main"
API_URL="https://huggingface.co/api/datasets/shangxd/vidor/tree/main?recursive=true"

mkdir -p "$TARGET_DIR"

# ────────────────────────── 2. make fresh file list ──────────────────────────
echo "Fetching file list from Hugging Face Hub …"
curl -s "$API_URL" | jq -r '.[].path' > filelist.txt

# ────────────────────────── 3. download every file ───────────────────────────
while read -r file; do
    [[ -z "$file" ]] && continue                     # skip empty lines

    dest="$TARGET_DIR/$file"
    echo "Fetching $file …"
    mkdir -p "$(dirname "$dest")"                    # recreate sub-dirs
    wget -q --show-progress -c "$BASE_URL/$file" \
         -O "$dest"                                  # resume-capable (-c)
done < filelist.txt
echo "Download stage complete."

# ────────────────────────── 4. unzip, then delete the zips ───────────────────
echo "Extracting ZIP archives into $TARGET_DIR …"
find "$TARGET_DIR" -name '*.zip' | while read -r zip; do
    echo "  • $(basename "$zip")"
    unzip -qn "$zip" -d "$TARGET_DIR"                # -n = never overwrite
    rm -f "$zip"
done

echo "All done!  Everything is now under:  $TARGET_DIR"
