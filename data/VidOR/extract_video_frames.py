import os
import subprocess
import json
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────────
FFMPEG_PATH  = '/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffmpeg'
FFPROBE_PATH = '/scratch/project_462000938/vidor/ffmpeg-3.3.4/bin-linux/ffprobe'

VIDEO_ROOT   = '/scratch/project_462000938/vidor/dataset/video'
OUTPUT_ROOT  = '/scratch/project_462000938/vidor/dataset/extracted_frames'
ANNO_ROOTS   = [
    '/scratch/project_462000938/vidor/dataset/training',
    '/scratch/project_462000938/vidor/dataset/validation'
]
# ------------------------------------------------------------------------

def find_annotation_file(folder_name, video_id):
    for root in ANNO_ROOTS:
        anno = os.path.join(root, folder_name, f"{video_id}.json")
        if os.path.isfile(anno):
            return anno
    return None

def load_annotation(path):
    with open(path, 'r') as f:
        return json.load(f)

# collect every .mp4 (so tqdm shows a total)
video_files = [
    (folder, fname)
    for folder in sorted(os.listdir(VIDEO_ROOT))
    if os.path.isdir(os.path.join(VIDEO_ROOT, folder))
    for fname in os.listdir(os.path.join(VIDEO_ROOT, folder))
    if fname.endswith('.mp4')
]

for folder_name, filename in tqdm(video_files, desc="Processing videos"):
    folder_path = os.path.join(VIDEO_ROOT, folder_name)
    video_path  = os.path.join(folder_path, filename)
    video_id    = os.path.splitext(filename)[0]

    # ── locate annotation -------------------------------------------------
    anno_path = find_annotation_file(folder_name, video_id)
    if not anno_path:
        print(f"⚠️  Annotation not found for {video_id}, skipping.")
        continue
    anno = load_annotation(anno_path)

    fps          = int(round(anno.get('fps', 30.0)))  # treat as integral
    frame_count  = anno['frame_count']
    trajectories = anno['trajectories']
    relations    = anno.get('relation_instances', [])

    # ── output dirs -------------------------------------------------------
    out_dir   = os.path.join(OUTPUT_ROOT, folder_name, video_id)
    os.makedirs(out_dir, exist_ok=True)

    # if *any* jpg already there we assume the video is done
    if any(n.endswith('.jpg') for n in os.listdir(out_dir)):
        print(f"✅ Frames already extracted for {video_id}, skipping.")
        continue

    print(f"🎞️  {video_id}: saving frame 0 and every {fps}-th frame "
          f"(~1 fps from {anno['fps']:.2f} fps source)")

    # ── determine which frames to keep ------------------------------------
    keep_fids = list(range(0, frame_count, fps))
    if keep_fids[-1] != frame_count - 1:          # always include last frame
        keep_fids.append(frame_count - 1)

    # ── helper to extract one frame via ffmpeg ----------------------------
    def save_frame(fid, img_path):
        # select='eq(n\,fid)' picks exactly that frame
        cmd = [
            FFMPEG_PATH,
            '-loglevel', 'error',
            '-i', video_path,
            '-vf', f"select='eq(n\\,{fid})'",
            '-vframes', '1',
            img_path
        ]
        subprocess.run(cmd, check=True)

    # ── iterate over selected frames --------------------------------------
    for fid in keep_fids:
        img_name = f"frame_{fid:04d}.jpg"
        ann_name = f"ann_{fid:04d}.json"
        img_path = os.path.join(out_dir, img_name)
        ann_path = os.path.join(out_dir, ann_name)

        # 1. extract & save the frame
        try:
            save_frame(fid, img_path)
        except subprocess.CalledProcessError:
            print(f"❌ ffmpeg failed for {video_id} frame {fid}")
            continue

        # 2. slice annotation for this frame
        frame_boxes = trajectories[fid] if fid < len(trajectories) else []
        frame_rels  = [
            r for r in relations
            if r['begin_fid'] <= fid < r['end_fid']
        ]
        per_frame_ann = {
            "video_id": video_id,
            "frame_id": fid,
            "width":  anno['width'],
            "height": anno['height'],
            "objects": frame_boxes,
            "relations": frame_rels
        }
        with open(ann_path, 'w') as f:
            json.dump(per_frame_ann, f, indent=2)

print("✅ All done.")
