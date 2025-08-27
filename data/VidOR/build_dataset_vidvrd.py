#!/usr/bin/env python3
"""
Build a JSONL index of fixed-length, fixed-stride clips that contain at least
one object track appearing in **all** frames of the clip.

• Works with both **VIDOR** and **VIDVRD** annotation formats.
• Assumes one extracted JPEG frame ≙ one second of video.
• Output line format:
      {"video": "<relative/folder>.mp4", "frames": [f0, f1, …, fN]}
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional
import json, random, subprocess
from tqdm import tqdm

# ---------------------------------------------------------------------
# Optional ffprobe helpers (unused by the clip-index logic itself)
# ---------------------------------------------------------------------
FFPROBE = "/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffprobe"
FFMPEG  = "/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffmpeg"


def video_metadata_ffprobe(video_path: Path) -> Tuple[float, float, int]:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=duration,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    proc   = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(proc.stdout)["streams"][0]

    dur = float(stream.get("duration", 0.0))
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if den != "0" else 0.0
    n_frames = int(stream.get("nb_read_frames", 0) or round(dur * fps) if fps else 0)
    return dur, fps, n_frames


def video_duration_seconds(p: Path) -> int:
    return int(video_metadata_ffprobe(p)[0])


# ---------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------
def load_tracks(json_file: Path) -> Dict[int, Set[int]]:
    """
    Parse an annotation file (VIDOR or VIDVRD) and return
        {frame_id → set(track_ids_visible_in_that_frame)}.
    Frames with no non-generated boxes are omitted.
    """
    data = json.loads(json_file.read_text())
    tracks: Dict[int, Set[int]] = {}

    # ---- VIDOR style ---------------------------------------------------
    # {'trajectories': [list(boxes_f0), list(boxes_f1), …]}
    if isinstance(data, dict) and "trajectories" in data:
        for fid, boxes in enumerate(data["trajectories"]):
            if boxes:
                tracks[fid] = {box["tid"] for box in boxes}

    # ---- VIDVRD style --------------------------------------------------
    # Either a list of per-frame dicts or dict{'annotations': [...]}
    else:
        frames = data if isinstance(data, list) else data.get("annotations", [])
        for frame in frames:
            fid  = int(frame["frame_id"])  # e.g. 0, 30, 60 …
            tids = {
                obj["tid"]
                for obj in frame.get("objects", [])
                if not obj.get("generated", 0)
            }
            if tids:
                tracks[fid] = tids

    return tracks


def has_common_track(tracks: Dict[int, Set[int]], frames: List[int]) -> bool:
    """True iff there is at least one track-id visible in *all* frames."""
    common: Optional[Set[int]] = None
    for f in frames:
        tids = tracks.get(f, set())
        common = tids if common is None else common & tids
        if not common:
            return False
    return True


# ---------------------------------------------------------------------
# Build the clip index
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------
    # 0 / Configuration
    # ---------------------------------------------------------------
    DATASET = "vidvrd"  # "vidor" or "vidvrd"

    if DATASET == "vidor":
        anno_roots = [
            Path("/scratch/project_462000938/vidor/dataset/training"),
            Path("/scratch/project_462000938/vidor/dataset/validation"),
        ]
        frames_root = Path("/scratch/project_462000938/vidor/dataset/extracted_frames")
    else:  # VIDVRD
        anno_roots = [
            Path("/scratch/project_462000938/vidvrd/vidvrd-ann/training"),
            # Path("/scratch/project_462000938/vidvrd/vidvrd-ann/validation"),
        ]
        frames_root = Path("/scratch/project_462000938/vidvrd/extracted_frames")

    N = 2                      # frames per clip (clip length)
    STRIDES = [1, 3]           # seconds between consecutive frames
    MIN_SEC = 1                # reject videos shorter than this

    C_MIN, C_MAX = 10, 30      # clip count lower / upper bounds per video

    # ---------------------------------------------------------------
    # 1 / Collect annotation files
    # ---------------------------------------------------------------
    json_files = [p for root in anno_roots for p in root.rglob("*.json")]
    desc       = f"Scanning {DATASET.upper()} videos"

    # ---------------------------------------------------------------
    # 2 / Build index for each stride
    # ---------------------------------------------------------------
    for stride_s in STRIDES:
        index: List[Tuple[Path, List[int]]] = []   # (relative_folder, frame_ids)

        with tqdm(json_files, desc=f"{desc} (stride={stride_s}s)") as pbar:
            total_clips = 0

            for ann_path in pbar:
                vid_id = ann_path.stem

                # Locate extracted frame folder
                if DATASET == "vidor":
                    subdir       = ann_path.parent.name         # e.g. "0000"
                    frame_folder = frames_root / subdir / vid_id
                else:  # vidvrd
                    frame_folder = frames_root / vid_id

                if not frame_folder.is_dir():
                    continue

                # ---------------------------------------------------
                # 2.1 Load frame list & tracks
                # ---------------------------------------------------
                frame_files = sorted(
                    frame_folder.glob("frame_*.jpg"),
                    key=lambda p: int(p.stem.split("_")[-1]),
                )
                frame_ids = [int(p.stem.split("_")[-1]) for p in frame_files]

                n_frames = len(frame_files)
                dur_sec  = n_frames          # 1 extracted frame ≙ 1 s
                if n_frames < N or dur_sec < MIN_SEC:
                    continue

                tracks = load_tracks(ann_path)

                # ---------------------------------------------------
                # 2.2 Randomly sample up to C_dyn clips
                # ---------------------------------------------------
                stride_f   = stride_s
                last_start = max(0, n_frames - (N - 1) * stride_f - 1)
                t0_candidates = list(range(last_start))
                random.shuffle(t0_candidates)

                frames_per_clip   = (N - 1) * stride_f + 1
                max_clips_by_len  = n_frames // frames_per_clip
                C_dyn             = max(C_MIN, min(max_clips_by_len, C_MAX))

                clips_found, frames_used = 0, set()
                for t0 in t0_candidates:
                    frames = [frame_ids[t0 + k * stride_f] for k in range(N)]

                    if any(f in frames_used for f in frames):
                        continue
                    if has_common_track(tracks, frames):
                        rel_folder = frame_folder.relative_to(frames_root)
                        index.append((rel_folder, frames))
                        clips_found  += 1
                        total_clips  += 1
                        frames_used.update(frames)
                        pbar.set_postfix(total_clips=total_clips)
                    if clips_found == C_dyn:
                        break

        # -----------------------------------------------------------
        # 3 / Save index
        # -----------------------------------------------------------
        out_file = Path(f"{DATASET}_clip_index_{stride_s}s.jsonl")
        with out_file.open("w") as f:
            for folder_rel, frames in index:
                line = {"video": str(folder_rel) + ".mp4", "frames": frames}
                f.write(json.dumps(line) + "\n")

        print(
            f"\n✅ Built {len(index)} clips from {len(json_files)} videos "
            f"(stride={stride_s}s)."
        )
        print("First 3 entries:", index[:3])
        print(f"✅ Saved to {out_file}\n")
