#!/usr/bin/env python3
# make_zip_vidor.py
#
#
# --------------------------------------------------------------------------------

import argparse, json, zipfile
from pathlib import Path
from tqdm import tqdm   # pip install tqdm if needed

def build_zip(index_path: Path,
              frames_root: Path,
              anno_roots: list[Path],
              out_zip: Path):

    # ─── open zip for writing ────────────────────────────────────────────────────
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as z:

        # 1) drop the raw index inside the archive
        z.write(index_path, arcname="index.jsonl")

        written_annos = set()
        written_imgs  = set()
        # 2) iterate over every video listed in the index
        with index_path.open("r") as f:
            lines = f.readlines()

        for ln in tqdm(lines, desc=f"Zipping to {out_zip.name}", unit="video"):
                rec = json.loads(ln)
                path = rec["video"]          # e.g. 0000/2401075277.mp4
                cat = path[:4]                  # e.g. 0000
                vid = path[5:-4]                # e.g. 2401075277
                
                # ─── frames ────────────────────────────────────────────────────
                frame_dir = frames_root / cat / vid
                if not frame_dir.is_dir():
                    print(f"[WARN] missing frame dir: {frame_dir}")
                else:
                    for fi in rec["frames"]:                # e.g. [927, 957]
                        # VidOR frames are named frame_0000.jpg … frame_9999.jpg
                        # zero-pad to 4 digits; adjust if your naming is different.
                        cand = frame_dir / f"frame_{fi:04d}.jpg"
                        if not cand.is_file():              # fall-back: try 5-digit padding
                            cand = frame_dir / f"frame_{fi:05d}.jpg"
                        if not cand.is_file():
                            print(f"[WARN] missing frame {fi} for {vid}")
                            continue

                        rel_img = f"images/{cat}/{vid}/{cand.name}"
                        if rel_img not in written_imgs:
                            z.write(cand, arcname=rel_img)
                            written_imgs.add(rel_img)
                        

                # ─── annotation ──────────────────────────────────────────────
                anno_file = next((root / cat / f"{vid}.json"
                                  for root in anno_roots
                                  if (root / cat / f"{vid}.json").is_file()),
                                 None)
                if anno_file is None:
                    print(f"[WARN] missing annotation for {vid}")
                else:
                    rel_anno = f"annotations/{cat}/{vid}.json"
                    if rel_anno not in written_annos:      # ← check before writing
                        z.write(anno_file, arcname=rel_anno)
                        written_annos.add(rel_anno)

    print(f"\n✅  Wrote {out_zip} ({out_zip.stat().st_size/1e6:.1f} MB)")

# ─── CLI glue ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index",   required=True, type=Path)
    p.add_argument("--frames",  required=True, type=Path)
    p.add_argument("--anno",    required=True, type=Path, nargs="+")
    p.add_argument("--output",  required=True, type=Path)
    args = p.parse_args()

    build_zip(args.index, args.frames, args.anno, args.output)

"""
python make_zip_vidor.py \
       --index   /users/karasari/Object-Level-Self-Supervised-Learning/VidOR/36k_25s.jsonl \
       --frames  /scratch/project_462000938/vidor/dataset/extracted_frames \
       --anno    /scratch/project_462000938/vidor/dataset/training /scratch/project_462000938/vidor/dataset/validation \
       --output /scratch/project_462000938/odis/dest/vidor_36Kclips_25s.zip
"""
# next one : 25s