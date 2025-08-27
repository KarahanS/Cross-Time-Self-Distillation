#!/usr/bin/env python3
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tqdm.auto import tqdm

# ---------- CONFIG ----------------------------------------------------------- 
fps              = 60            # frames per second
interval_seconds = 1             # seconds between frame pairs
interval_frames  = fps * interval_seconds

input_dir = Path("/scratch/project_462000938/wt_venice/extracted_frames")
zip_path  = Path("/scratch/project_462000938/wt_venice/venice_1sec.zip")
# -----------------------------------------------------------------------------


# Sorted list of all *.jpg frames
frames       = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".jpg")
total_frames = len(frames)

# Each clip consumes 2·interval_frames unique images
step       = interval_frames * 2
max_clips  = (total_frames - interval_frames) // step          # progress-bar length

clip_idx = 0
i = 0

with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf, \
     tqdm(total=max_clips, desc="Writing clips to zip") as pbar:

    while i + interval_frames < total_frames:
        f1 = frames[i]
        f2 = frames[i + interval_frames]

        # Inside the zip we mimic the old structure: <clip_idx>/<filename>.jpg
        zf.write(f1, arcname=f"{clip_idx}/{f1.name}")
        zf.write(f2, arcname=f"{clip_idx}/{f2.name}")

        # advance — skip both frames that formed this clip
        i += step
        clip_idx += 1
        pbar.update(1)

print(f"🗜️  Done. Created {clip_idx} clips directly in '{zip_path}'.")
