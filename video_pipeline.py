from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from config import (
    VIDEO_SAMPLE_EVERY_N_FRAMES,
    VIDEO_MAX_FRAMES,
    VIDEO_FAKE_THRESHOLD,
)
from image_pipeline import DeepfakeImagePredictor


class VideoDeepfakePredictor:
    def __init__(self, image_predictor: DeepfakeImagePredictor):
        self.image_predictor = image_predictor

        # Use OpenCV's built-in Haar cascade path (more reliable than hardcoding macOS path)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    def _extract_sampled_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % VIDEO_SAMPLE_EVERY_N_FRAMES == 0:
                frames.append(frame)

            frame_idx += 1

            if len(frames) >= VIDEO_MAX_FRAMES:
                break

        cap.release()
        return frames

    def _detect_main_face(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return None

        # pick largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        pad_x = int(0.15 * w)
        pad_y = int(0.15 * h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_bgr.shape[1], x + w + pad_x)
        y2 = min(frame_bgr.shape[0], y + h + pad_y)

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def predict_video(self, video_path: str):
        if video_path is None or not Path(video_path).exists():
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "p_real": 0.0,
                "p_fake": 0.0,
                "frames_processed": 0,
                "faces_found": 0,
                "fake_frames": 0,
                "real_frames": 0,
                "details": f"Video path not found: {video_path}",
            }

        frames = self._extract_sampled_frames(video_path)

        if not frames:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "p_real": 0.0,
                "p_fake": 0.0,
                "frames_processed": 0,
                "faces_found": 0,
                "fake_frames": 0,
                "real_frames": 0,
                "details": "No frames could be read from the video.",
            }

        frame_results = []
        faces_found = 0

        for frame in frames:
            face_img = self._detect_main_face(frame)

            # If no face found in frame, skip it
            if face_img is None:
                continue

            faces_found += 1
            result = self.image_predictor.predict_pil(face_img)
            frame_results.append(result)

        if not frame_results:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "p_real": 0.0,
                "p_fake": 0.0,
                "frames_processed": len(frames),
                "faces_found": 0,
                "fake_frames": 0,
                "real_frames": 0,
                "details": "Frames were read, but no faces were detected in sampled frames.",
            }

        avg_p_real = float(np.mean([r["p_real"] for r in frame_results]))
        avg_p_fake = float(np.mean([r["p_fake"] for r in frame_results]))

        fake_frames = sum(1 for r in frame_results if r["label"] == "Fake")
        real_frames = sum(1 for r in frame_results if r["label"] == "Real")

        final_label = "Fake" if avg_p_fake >= VIDEO_FAKE_THRESHOLD else "Real"
        final_confidence = avg_p_fake if final_label == "Fake" else avg_p_real

        return {
            "label": final_label,
            "confidence": final_confidence,
            "p_real": avg_p_real,
            "p_fake": avg_p_fake,
            "frames_processed": len(frames),
            "faces_found": faces_found,
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "details": f"Processed {len(frames)} sampled frames and detected faces in {faces_found} frames.",
        }