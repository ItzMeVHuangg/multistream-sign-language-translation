import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import mediapipe as mp
import yaml
import concurrent.futures

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global variables for workers
worker_pose_lm = None
worker_hand_lm = None

def init_worker():
    global worker_pose_lm, worker_hand_lm
    
    p_task = Path(__file__).parent / "pose_landmarker_lite.task"
    h_task = Path(__file__).parent / "hand_landmarker.task"
    
    base_pose = python.BaseOptions(model_asset_path=str(p_task))
    options_pose = vision.PoseLandmarkerOptions(
        base_options=base_pose,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5)
    worker_pose_lm = vision.PoseLandmarker.create_from_options(options_pose)

    base_hands = python.BaseOptions(model_asset_path=str(h_task))
    options_hands = vision.HandLandmarkerOptions(
        base_options=base_hands,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5)
    worker_hand_lm = vision.HandLandmarker.create_from_options(options_hands)

def process_video(args):
    v_dir, out_file = args
    if out_file.exists():
        return True

    frame_files = sorted(v_dir.glob("*.png")) or sorted(v_dir.glob("*.jpg"))
    if not frame_files:
        return False

    all_kpts = []
    for fp in frame_files:
        with Image.open(fp) as img:
            pil_img = img.convert("RGB")
        img_np = np.array(pil_img)
        img_data = np.ascontiguousarray(img_np)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_data)
        
        pose_res = worker_pose_lm.detect(mp_img)
        hands_res = worker_hand_lm.detect(mp_img)
        
        kp = []
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                kp.append([lm.x, lm.y, lm.z])
        else:
            kp.extend([[0, 0, 0]] * 33)
        
        hands_by_side = {"Left": None, "Right": None}
        if hands_res.hand_landmarks:
            handedness = getattr(hands_res, "handedness", None) or []
            for i, hand_lm in enumerate(hands_res.hand_landmarks[:2]):
                side = None
                if i < len(handedness) and handedness[i]:
                    cat = handedness[i][0]
                    side = getattr(cat, "category_name", None) or getattr(cat, "display_name", None)
                if side not in hands_by_side:
                    side = "Left" if hands_by_side["Left"] is None else "Right"
                hands_by_side[side] = hand_lm

        hand_kp = []
        for side in ("Left", "Right"):
            hand_lm = hands_by_side[side]
            if hand_lm is None:
                hand_kp.extend([[0, 0, 0]] * 21)
                continue
            wx, wy, wz = hand_lm[0].x, hand_lm[0].y, hand_lm[0].z
            for lm in hand_lm:
                hand_kp.append([lm.x - wx, lm.y - wy, lm.z - wz])
        
        kp.extend(hand_kp)
        all_kpts.append(np.array(kp).flatten())
        
    np.save(str(out_file), np.array(all_kpts, dtype=np.float32))
    return True

def extract_keypoints_for_dataset(cfg):
    frames_root = Path(cfg["paths"]["frames_root"])
    out_root = Path(cfg["paths"]["mediapipe_kpts_root"])

    tasks = []
    for split in ["train", "dev", "test"]:
        split_dir = frames_root / split
        out_split_dir = out_root / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        if not split_dir.exists():
            continue

        video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for v_dir in video_dirs:
            out_file = out_split_dir / f"{v_dir.name}.npy"
            tasks.append((v_dir, out_file))

    print(f"Total videos to extract: {len(tasks)}")
    
    # Restrict max workers to respect Slurm cpu-per-task environment, avoiding crashes 
    allowed_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    max_workers = min(allowed_cpus, 16)
    print(f"Running extraction using {max_workers} concurrent separate processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        list(tqdm(executor.map(process_video, tasks), total=len(tasks), desc="Extracting MediaPipe Keypoints"))
    print("Extraction Complete!")

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    extract_keypoints_for_dataset(cfg)
