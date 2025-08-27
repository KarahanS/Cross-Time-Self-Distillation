#!/usr/bin/env python3
# check_ffprobe_vs_annotation_live.py
#
# Stream mismatches to the console as soon as they’re detected.

import os
import subprocess
import json
from tqdm import tqdm

# ───── dataset-specific paths ────────────────────────────────────────────
FFPROBE_PATH = '/scratch/project_462000938/vidor/dataset/ffmpeg-3.3.4/bin-linux/ffprobe'

VIDEO_ROOT   = '/scratch/project_462000938/vidor/dataset/video'
ANNO_ROOTS   = [
    '/scratch/project_462000938/vidor/dataset/training',
    '/scratch/project_462000938/vidor/dataset/validation'
]
# ─────────────────────────────────────────────────────────────────────────

def ffprobe_frame_count(video_path):
    """Return the number of frames, or None if it can’t be read."""
    # 1) Slow-but-reliable nb_read_frames
    cmd = [
        FFPROBE_PATH, '-v', 'error', '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if out.isdigit():
        return int(out)

    # 2) Fallback nb_frames
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None

def find_annotation_file(folder, vid):
    for root in ANNO_ROOTS:
        path = os.path.join(root, folder, f"{vid}.json")
        if os.path.isfile(path):
            return path
    return None

def collect_mp4s():
    for folder in sorted(os.listdir(VIDEO_ROOT)):
        fpath = os.path.join(VIDEO_ROOT, folder)
        if not os.path.isdir(fpath):
            continue
        for name in os.listdir(fpath):
            if name.endswith('.mp4'):
                yield folder, name

def main():
    total = ok = 0
    mismatches = []

    mp4s = list(collect_mp4s())
    for folder, mp4 in tqdm(mp4s, desc="Verifying"):
        total += 1
        vid      = os.path.splitext(mp4)[0]
        vpath    = os.path.join(VIDEO_ROOT, folder, mp4)
        annopath = find_annotation_file(folder, vid)

        if not annopath:
            msg = f"{vid}: annotation JSON not found"
            mismatches.append(msg)
            tqdm.write(msg)
            continue

        with open(annopath, 'r') as f:
            declared = json.load(f).get('frame_count')

        actual = ffprobe_frame_count(vpath)
        if actual is None:
            msg = f"{vid}: ffprobe could not read frame count"
            mismatches.append(msg)
            tqdm.write(msg)
        elif actual != declared:
            msg = f"{vid}: declared={declared}, ffprobe={actual}"
            mismatches.append(msg)
            tqdm.write(msg)
        else:
            ok += 1

    # ─ summary ───────────────────────────────────────────────────────────
    print("\n──────── Summary ────────")
    print(f"Videos checked : {total}")
    print(f"Matches        : {ok}")
    print(f"Mismatches     : {len(mismatches)}")

    if mismatches:
        with open("mismatch_report.txt", 'w') as f:
            f.write("\n".join(mismatches))
        print("⚠️  Full list saved to mismatch_report.txt")

if __name__ == "__main__":
    main()
