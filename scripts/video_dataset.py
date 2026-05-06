# scripts/video_dataset.py
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def _is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXTS

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def uniform_frame_indices(n_total: int, n_samples: int) -> List[int]:
    if n_total <= 0:
        return [0] * n_samples
    if n_total <= n_samples:
        # repeat if video shorter than required samples
        idx = list(range(n_total))
        while len(idx) < n_samples:
            idx.append(idx[-1])
        return idx[:n_samples]
    # uniform sampling
    return np.linspace(0, n_total - 1, n_samples).round().astype(int).tolist()

def read_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = uniform_frame_indices(n_total, num_frames)

    frames = []
    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            # fallback: try reading sequentially
            ok, frame = cap.read()
        if frame is None:
            # last resort: create a blank frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames

class VideoFolderDataset(Dataset):
    """
    Expects:
      data_dir/
        real/  (videos)
        fake/  (videos)

    Returns:
      embeddings sequence will be produced later by CNN.
      Here we return frames as tensors: [T, 3, H, W]
    """
    def __init__(self, data_dir: str, num_frames: int = 16, img_size: int = 224):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.img_size = img_size

        self.samples: List[Tuple[str, int]] = []
        for label_name, y in [("real", 0), ("fake", 1)]:
            folder = self.data_dir / label_name
            if not folder.exists():
                continue
            for p in folder.rglob("*"):
                if p.is_file() and _is_video(p):
                    self.samples.append((str(p), y))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No videos found under {data_dir}. Expected real/ and fake/ with video files."
            )

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        # Resize and convert to float tensor [3,H,W] normalized to 0..1
        frame = cv2.resize(frame_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # HWC -> CHW
        return frame

    def __getitem__(self, idx: int):
        video_path, y = self.samples[idx]
        frames = read_video_frames(video_path, self.num_frames)
        x = torch.stack([self._preprocess(fr) for fr in frames], dim=0)  # [T,3,H,W]
        return x, torch.tensor(y, dtype=torch.long), video_path
