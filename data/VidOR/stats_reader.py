import json
import sys

def print_video_statistics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Basic video information
    video_id = data.get('video_id', 'Unknown')
    video_path = data.get('video_path', 'Unknown')
    frame_count = data.get('frame_count', 0)
    fps = data.get('fps', 0.0)
    width = data.get('width', 0)
    height = data.get('height', 0)

    # Subjects/objects
    subjects = data.get('subject/objects', [])
    num_subjects = len(subjects)

    # Trajectories
    trajectories = data.get('trajectories', [])
    num_trajectories = sum(len(frame) for frame in trajectories)
    num_frames_with_trajectories = len(trajectories)

    # Generated vs manually labeled boxes
    manually_labeled = sum(1 for frame in trajectories for box in frame if box.get('generated') == 0)
    generated = num_trajectories - manually_labeled

    # Relation instances
    relation_instances = data.get('relation_instances', [])
    num_relations = len(relation_instances)

    print(f"📼 Video ID: {video_id}")
    print(f"📁 Video Path: {video_path}")
    print(f"🎞️ Total Frames: {frame_count}")
    print(f"⏱️ FPS: {fps:.2f}")
    print(f"📏 Resolution: {width}x{height}")
    print(f"🧍 Number of Subjects/Objects: {num_subjects}")
    print(f"📊 Total Bounding Boxes: {num_trajectories}")
    print(f"📸 Frames with Trajectories: {num_frames_with_trajectories}")
    print(f"✍️ Manually Labeled Boxes: {manually_labeled}")
    print(f"⚙️ Generated Boxes: {generated}")
    print(f"🔗 Visual Relations Annotated: {num_relations}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stats_reader.py <path_to_json>")
    else:
        print_video_statistics(sys.argv[1])
