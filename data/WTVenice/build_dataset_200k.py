#!/usr/bin/env python3
"""
Build 100 000 two–frame clips from a 60 fps frame folder.

Rule
----
You may not pick both i and i+60 as clip starts, but any larger overlap is OK.
We enforce that by taking starts only from the first half of every 120-frame
window:

                     0 … 59   ← allowed
    window 0  :  |-------------|-------------|
                     60 …119   ← forbidden
    window 1  :                |-------------|-------------|
                               120…179  allowed
                     ... etc.

That guarantees:
    • no pair of starts differs by 60 frames
    • start reuse rate ≈ ½, so with 396 000 frames we still have ≈198 000
      eligible start positions — plenty to sample 100 000 from.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tqdm.auto import tqdm
import math

# ─── USER SETTINGS ───────────────────────────────────────────────────────────
FPS        = 60                                 # frames per second
INTERVAL   = FPS                                # 1-second gap → 60 frames
WINDOW     = INTERVAL * 2                       # 120-frame block
N_CLIPS    = 100_000                            # want 200k images
INPUT_DIR  = Path("/scratch/project_462000938/wt_venice/extracted_frames")
ZIP_OUT    = Path("/scratch/project_462000938/wt_venice/venice_1sec_100kclips.zip")
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_list(total: int) -> list[int]:
    """Return every index i where (i % 120) < 60  AND  i+60 is still in range."""
    max_start = total - INTERVAL - 1
    return [i for i in range(max_start + 1) if (i % WINDOW) < INTERVAL]

def pick_uniform(starts: list[int], n: int) -> list[int]:
    """Pick n indices evenly from a sorted candidate list."""
    stride = len(starts) / n
    return [starts[ math.floor(i * stride) ] for i in range(n)]

def main() -> None:
    frames = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".jpg")
    total_frames = len(frames)
    if total_frames < INTERVAL + 1:
        raise RuntimeError("Video too short for even one clip.")

    candidates = build_candidate_list(total_frames)
    if len(candidates) < N_CLIPS:
        raise RuntimeError(
            f"Only {len(candidates):,} eligible starts (< 60-apart rule) "
            f"but {N_CLIPS:,} requested."
        )

    starts = pick_uniform(candidates, N_CLIPS)

    with ZipFile(ZIP_OUT, "w", compression=ZIP_DEFLATED) as zf, \
         tqdm(total=N_CLIPS, desc="Writing clips") as bar:
        for clip_idx, s in enumerate(starts):
            zf.write(frames[s],            arcname=f"{clip_idx}/{frames[s].name}")
            zf.write(frames[s + INTERVAL], arcname=f"{clip_idx}/{frames[s + INTERVAL].name}")
            bar.update(1)

    print(f"✅  Wrote {N_CLIPS:,} clips → {N_CLIPS*2:,} images → {ZIP_OUT}")

if __name__ == "__main__":
    main()
