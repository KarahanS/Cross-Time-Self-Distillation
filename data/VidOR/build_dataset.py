from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional
import json, random, subprocess
from tqdm import tqdm

# ---------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------
FFPROBE = "/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffprobe"
FFMPEG ="/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffmpeg"

def video_metadata_ffprobe(video_path: Path) -> Tuple[float, float, int]:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=duration,avg_frame_rate,nb_read_frames",
        "-of", "json", str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(proc.stdout)["streams"][0]

    dur = float(stream.get("duration", 0.0))
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if den != "0" else 0.0
    n_frames = int(stream.get("nb_read_frames", 0) or round(dur * fps) if fps else 0)
    return dur, fps, n_frames

def video_duration_seconds(p: Path) -> int:
    return int(video_metadata_ffprobe(p)[0])

# ---------------------------------------------------------------------
# helpers for paths / tracks
# ---------------------------------------------------------------------
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

def all_videos(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)

def find_annotation(video_path: Path, anno_roots: Iterable[Path]) -> Optional[Path]:
    folder, stem = video_path.parts[-2], video_path.stem
    for root in anno_roots:
        cand = root / folder / f"{stem}.json"
        if cand.is_file():
            return cand
    return None

def load_tracks(json_file: Path) -> Dict[int, Set[int]]:
    ann = json.loads(json_file.read_text())
    tracks: Dict[int, Set[int]] = {}
    for fid, boxes in enumerate(ann["trajectories"]):
        if boxes:
            tracks[fid] = {b["tid"] for b in boxes}
    return tracks

def has_common_track(tracks: Dict[int, Set[int]], frames: List[int]) -> bool:
    """True iff there is at least one track-id visible in *all* frames."""
    common: Set[int] | None = None
    for f in frames:
        tids = tracks.get(f, set())
        common = tids if common is None else common & tids
        if not common:
            return False
    return True

# ---------------------------------------------------------------------
# build the clip index
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# build the clip index  (no ffprobe, use pre-extracted frames)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    anno_roots = [
        Path("/scratch/project_462000938/vidor/dataset/training"),
        Path("/scratch/project_462000938/vidor/dataset/validation"),
    ]
    frames_root = Path("/scratch/project_462000938/vidor/dataset/extracted_frames")

    C, N, stride_s, min_sec = 10, 2, 1, 10      # #clips per video, clip length, stride(sec), min length
    

    json_files = list(anno.rglob("*.json") for anno in anno_roots)
    json_files = [p for sub in json_files for p in sub]
    
    for stride_s in (15, 20, 25):
        with tqdm(json_files, desc="Scanning videos") as pbar:
            index: List[Tuple[Path, List[int]]] = []     # (frame-folder, [frame indices])
                              # keep at least a couple (e.g. 5)

            total_clips = 0
            for ann_path in pbar:
                meta   = json.loads(ann_path.read_text())
                vid_id = meta["video_id"]
                fps = 1 # LUMI

                subdir       = ann_path.parent.name              # "0000"
                frame_folder = frames_root / subdir / vid_id     # .../0000/2401075277
                if not frame_folder.is_dir():
                    continue

                frame_files = sorted(frame_folder.glob("frame_*.jpg"),
                                    key=lambda p: int(p.stem.split("_")[-1]))
                frame_ids = [int(p.stem.split("_")[-1]) for p in frame_files]   # e.g. [1,2,3,…]
                
                
                n_frames = len(frame_files)
                dur_sec  = n_frames # because 1 frame = 1 s
                if n_frames < N or dur_sec < min_sec:
                    continue

                stride_f   = stride_s
                last_start = n_frames - (N - 1)*stride_f - 1        # inclusive upper bound
                t0_candidates = list(range(last_start))              # 0-based positions

                
                C_min, C_max = 10, 30        # tune these two numbers
                frames_per_clip = (N - 1)*stride_f + 1
                max_clips_by_len = n_frames // frames_per_clip          # how many *non-overlapping* clips fit
                C_dyn = min(max_clips_by_len, C_max)                    # C_max is a hard ceiling (e.g. 50)
                C_dyn = max(C_dyn, C_min) 
                
                
                tracks = load_tracks(ann_path)
                random.shuffle(t0_candidates)

                clips_found, frames_used = 0, set()
                for t0 in t0_candidates:
                    # positions → real ids
                    frames = [frame_ids[t0 + k*stride_f] for k in range(N)]

                    if any(f in frames_used for f in frames):
                        continue
                    if has_common_track(tracks, frames):
                        index.append((frame_folder.relative_to(frames_root), frames))
                        clips_found   += 1
                        total_clips   += 1
                        frames_used.update(frames)
                        pbar.set_postfix(clips=f"{total_clips}")
                    if clips_found == C_dyn:
                        break

        print(f"\n✅ Built {len(index)} clips from {len(json_files)} videos.")
        print("First 3 entries:", index[:3])

        # ---------------------------------------------------------------
        # save as JSONL (frame folder stored relative to frames_root)
        # ---------------------------------------------------------------
        out_file = Path(f"vidor_clip_index_{stride_s}s.jsonl")
        with out_file.open("w") as f:
            for folder_rel, frames in index:
                f.write(json.dumps({"video": str(folder_rel)+".mp4", "frames": frames}) + "\n")

        print(f"✅ Saved {len(index)} clips to {out_file}")
