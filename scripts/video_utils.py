import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class FrameExtractor:
    """
    Extracts frames from a video file at a specified interval.
    """
    def __init__(self, video_path, sample_rate=10):
        """
        :param video_path: Path to the video file.
        :param sample_rate: Extract every Nth frame.
        """
        self.video_path = str(video_path)
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def __iter__(self):
        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if count % self.sample_rate == 0:
                # Convert BGR (OpenCV) to RGB (PIL/Torch)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(frame_rgb)
            
            count += 1
    
    def __del__(self):
        if self.cap:
            self.cap.release()


class FaceDetector:
    """
    Detects faces in an image using MTCNN.
    """
    def __init__(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
        
        self.device = device
        # keep_all=False returns only the largest face (best for single-subject deepfakes)
        # select_largest=True ensures we focus on the main subject
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)

    def crop_face(self, img_pil):
        """
        Detects and returns the cropped face from a PIL image.
        Returns None if no face is found.
        """
        # MTCNN returns a tensor if return_prob=False (default) is not used appropriately or 
        # we can just use the forward pass which returns a cropped tensor directly.
        # But to keep it simple and compatible with our 'infer.py' transforms, 
        # let's get the box and crop manually or let MTCNN return the PIL crop.
        
        # MTCNN can return PIL image directly if we don't pass extract_face=False? 
        # Actually mtcnn(img) returns a tensor of the face.
        # To match our pipeline, we might want the PIL image back or just the tensor.
        # However, our infer.py expects a PIL image to apply its own transforms.
        
        # Let's get boxes and crop manually to be safe.
        boxes, _ = self.mtcnn.detect(img_pil)
        
        if boxes is None:
            return None
        
        # Take the first (largest) face
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # padding
        w, h = img_pil.size
        # slightly expand box
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        face = img_pil.crop((x1, y1, x2, y2))
        return face
