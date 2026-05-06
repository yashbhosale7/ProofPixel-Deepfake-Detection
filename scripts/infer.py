# scripts/infer.py
import argparse
from pathlib import Path
import os, json
import numpy as np
from io import BytesIO
import torch
import timm
from torchvision import transforms
from PIL import Image

# ---------------- device ----------------
def get_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --------------- paths ------------------
ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
SINGLE_DIR = ROOT / "data" / "infer" / "single"
BATCH_DIR  = ROOT / "data" / "infer" / "batch"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def list_images(folder: Path):
    return [p for ext in IMG_EXTS for p in folder.glob(ext)]

def newest_image(folder: Path):
    files = list_images(folder)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    return max(files, key=lambda p: p.stat().st_mtime)

def pick_checkpoint():
    for name in ("best.pt", "resnet50_best.pth", "best.pth", "model_best.pt"):
        p = CKPT_DIR / name
        if p.exists():
            return str(p)
    raise SystemExit(f"No checkpoint found in {CKPT_DIR}")

# --------------- model ------------------
def build_model(model_name: str, num_classes: int, ckpt_path: str, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def build_transform(model_name: str):
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    size = 224  # keep 224 for consistency
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ----------- simple forensic cues -----------
def _fft_ratio(img_pil: Image.Image) -> float:
    g = np.asarray(img_pil.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = max(2, min(cy, cx)//6)
    center = float(mag[cy-r:cy+r, cx-r:cx+r].sum())
    total  = float(mag.sum()) + 1e-6
    return float(1.0 - (center/total))

def _ela_score(img_pil: Image.Image, q: int = 90) -> float:
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=q, optimize=True)
    comp = Image.open(BytesIO(buf.getvalue())).convert("RGB")
    a = np.asarray(img_pil, dtype=np.int16)
    b = np.asarray(comp,   dtype=np.int16)
    diff = np.abs(a - b)
    return float(diff.mean())

# --------------- predict ----------------
CLASSES = ["fake", "real"]

def predict_one(model, tfm, device, path):
    if isinstance(path, (str, Path)):
        img = Image.open(path).convert("RGB")
    else:
        # Assume it's a PIL Image (from video crop)
        img = path.convert("RGB")
        path = "in-memory-frame"

    # network prob
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu() #activation function
    p_fake = float(probs[0].item())
    p_real = float(probs[1].item())
    # Use model output directly for label
    final_label = "real" if p_real > p_fake else "suspect_fake"
    final_conf  = p_real if p_real > p_fake else p_fake
    # Still compute ELA/FFT for info, but don't override label
    ela = float(_ela_score(img))
    fft = float(_fft_ratio(img))
    return {
        "file": str(path),
        "label": final_label,
        "confidence": final_conf,
        "p_real": float(p_real),
        "p_fake": float(p_fake),
        "ela": float(ela),
        "fft_ratio": float(fft)
    }

# --------------- video ------------------
from video_utils import FrameExtractor, FaceDetector

def predict_video(model, tfm, device, video_path: Path):
    """
    Predicts if a video is Real or Fake by aggregating frame predictions.
    """
    frame_ext = FrameExtractor(video_path, sample_rate=15) # Check every 15th frame
    detector = FaceDetector(device)
    
    results = []
    frames_processed = 0
    faces_found = 0
    
    for frame in frame_ext:
        frames_processed += 1
        face = detector.crop_face(frame)
        if face:
            faces_found += 1
            # reuse predict_one logic, but pass PIL image directly
            res = predict_one(model, tfm, device, face) # predict_one now handles PIL
            results.append(res)
    
    if not results:
        return {
            "file": str(video_path),
            "label": "unknown",
            "confidence": 0.0,
            "p_real": 0.0,
            "p_fake": 0.0,
            "details": f"Processed {frames_processed} frames using interval 15, found 0 faces."
        }
    
    # Aggregate
    # Simple average of probabilities
    avg_p_real = np.mean([r['p_real'] for r in results])
    avg_p_fake = np.mean([r['p_fake'] for r in results])
    
    # Vote
    fake_votes = sum(1 for r in results if r['label'] == 'fake' or r['label'] == 'suspect_fake')
    real_votes = len(results) - fake_votes
    
    final_label = "real" if avg_p_real > avg_p_fake else "FAKE"
    final_conf = avg_p_real if final_label == "real" else avg_p_fake
    
    return {
        "file": str(video_path),
        "label": final_label,
        "confidence": float(final_conf),
        "p_real": float(avg_p_real),
        "p_fake": float(avg_p_fake),
        "frames_processed": frames_processed,
        "faces_found": faces_found,
        "fake_frames": fake_votes,
        "real_frames": real_votes
    }


# --------------- main -------------------
def main(args):
    # defaults (no args needed)
    image_path = Path(args.image) if args.image else None
    folder_path = Path(args.folder) if args.folder else None
    ckpt_path = args.checkpoint or pick_checkpoint()
    model_name = args.model or "resnet50"
    # auto mode: prefer batch if images exist there; else newest in single
    if not image_path and not folder_path:
        batch_imgs = list_images(BATCH_DIR)
        if batch_imgs:
            folder_path = BATCH_DIR
        else:
            image_path = newest_image(SINGLE_DIR)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using model: {model_name}")
    tfm = build_transform(model_name)
    model = build_model(model_name, num_classes=len(CLASSES), ckpt_path=ckpt_path, device=device)
    results = []
    if folder_path:
        imgs = list_images(folder_path)
        if not imgs:
            raise SystemExit(f"No images found in {folder_path}")
        print(f"Found {len(imgs)} images in {folder_path}")
        for p in imgs:
            results.append(predict_one(model, tfm, device, p))
    else:
        print(f"Using image: {image_path}")
        results.append(predict_one(model, tfm, device, image_path))
    # print results
    for r in results:
        print(
            f"Prediction: {r['label']} ({r['confidence']*100:.1f}%)  "
            f"p_real={r['p_real']*100:.1f}%  p_fake={r['p_fake']*100:.1f}%  "
            f"ela={r['ela']:.2f} fft={r['fft_ratio']:.2f}  -  {Path(r['file']).name}"
        )
    # JSON summary (now JSON-serializable)
    print(json.dumps({"results": results}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to a single image (optional)")
    ap.add_argument("--folder", default=None, help="path to a folder of images (optional)")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path (optional)")
    ap.add_argument("--model", default="resnet50", help="timm model name")
    args = ap.parse_args()
    main(args)
