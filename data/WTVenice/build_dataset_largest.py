#!/usr/bin/env python3
"""
Make 2-frame, 1-second-apart clips without re-using any frame.

Example: 0-60, 1-61, …, 59-119   |  120-180, 121-181, …, 179-239  |  …
Total clips = total_frames / 2  (here: 195 000 for 390 000 frames).
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tqdm.auto import tqdm

# ---------- CONFIG -----------------------------------------------------------
fps              = 60                       # original video fps
interval_seconds = 1
interval_frames  = fps * interval_seconds   # 60

input_dir = Path("/scratch/project_462000938/wt_venice/extracted_frames")
zip_path  = Path("/scratch/project_462000938/wt_venice/venice_1sec.zip")
# -----------------------------------------------------------------------------

# Collect frames in strictly ascending order (frame_000000.jpg …)
frames = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".jpg")
total_frames = len(frames)
assert total_frames >= 2 * interval_frames, "Not enough frames for one clip."

# ------------------------------------------------------------------
# Pairing strategy:
#   block_size = 2*interval_frames  (here 120)
#   inside each block, index 0-59  →  60-119,  etc.
# ------------------------------------------------------------------
block_size     = 2 * interval_frames            # 120
clips_per_block = interval_frames               # 60
n_full_blocks   = total_frames // block_size
pairs_expected  = n_full_blocks * clips_per_block

# Handle a trailing “partial” block, if any.
remainder_start = n_full_blocks * block_size
remainder_len   = total_frames - remainder_start
extra_pairs     = max(0, remainder_len - interval_frames)
pairs_expected += extra_pairs

print(f"🎞️  Total frames           : {total_frames:,}")
print(f"📦 Clips to be created     : {pairs_expected:,}")
print(f"🗜️  Writing to             : {zip_path}")

with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf, \
     tqdm(total=pairs_expected, desc="Creating clips") as pbar:

    clip_idx = 0

    # ---- full 120-frame blocks ------------------------------------
    for blk in range(n_full_blocks):
        base = blk * block_size
        for off in range(interval_frames):                # 0 … 59
            f1 = frames[base + off]
            f2 = frames[base + off + interval_frames]     # +60

            zf.write(f1, arcname=f"{clip_idx:06d}/{f1.name}")
            zf.write(f2, arcname=f"{clip_idx:06d}/{f2.name}")

            clip_idx += 1
            pbar.update(1)

    # ---- partial tail block (if > interval_frames) ---------------
    if extra_pairs:
        base = remainder_start
        for off in range(extra_pairs):                     # 0 … extra_pairs-1
            f1 = frames[base + off]
            f2 = frames[base + off + interval_frames]

            zf.write(f1, arcname=f"{clip_idx:06d}/{f1.name}")
            zf.write(f2, arcname=f"{clip_idx:06d}/{f2.name}")

            clip_idx += 1
            pbar.update(1)

print(f"✅ Done. Stored {clip_idx:,} clips in '{zip_path}'.")
