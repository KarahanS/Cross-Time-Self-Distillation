#!/usr/bin/env python3
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

ZIP_PATH = Path("/scratch/project_462000938/wt_venice/venice_1sec.zip")  # ← adjust

def inspect_zip(zip_path: Path) -> None:
    with ZipFile(zip_path) as zf:
        # collect folder names "0/", "1/", …
        folders = [name.split("/", 1)[0] for name in zf.namelist() if name.endswith(".jpg")]
        counts  = Counter(folders)

    n_clips = len(counts)
    print(f"Found {n_clips:,} clip-folders in {zip_path.name}\n")

    wrong = {k: v for k, v in counts.items() if v != 2}
    if not wrong:
        print("✅ Every folder contains exactly 2 images.")
    else:
        print("⚠️  The following folders have an unexpected image count:")
        for k, v in sorted(wrong.items(), key=lambda x: int(x[0])):
            print(f"  clip {k:>6}  →  {v} images")

if __name__ == "__main__":
    inspect_zip(ZIP_PATH)
