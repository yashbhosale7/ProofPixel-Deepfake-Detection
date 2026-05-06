from pathlib import Path
from typing import Union
from io import BytesIO

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGE_RESIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_checkpoint_path,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_RESIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(model_name: str, num_classes: int, ckpt_path: Union[str, Path], device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state[new_key] = v

    model.load_state_dict(cleaned_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _fft_ratio(img_pil: Image.Image) -> float:
    g = np.asarray(img_pil.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = max(2, min(cy, cx) // 6)
    center = float(mag[cy - r:cy + r, cx - r:cx + r].sum())
    total = float(mag.sum()) + 1e-6
    return float(1.0 - (center / total))


def _ela_score(img_pil: Image.Image, q: int = 90) -> float:
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=q, optimize=True)
    comp = Image.open(BytesIO(buf.getvalue())).convert("RGB")
    a = np.asarray(img_pil, dtype=np.int16)
    b = np.asarray(comp, dtype=np.int16)
    diff = np.abs(a - b)
    return float(diff.mean())


class DeepfakeImagePredictor:
    def __init__(self, checkpoint_path: Union[str, Path, None] = None):
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path()
        self.transform = build_transform()
        self.model = build_model(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            ckpt_path=self.checkpoint_path,
            device=self.device,
        )

    def predict_pil(self, img: Image.Image):
        img = img.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].detach().cpu()

        p_fake = float(probs[0].item())
        p_real = float(probs[1].item())

        final_label = "Real" if p_real >= p_fake else "Fake"
        final_confidence = p_real if p_real >= p_fake else p_fake

        ela = _ela_score(img)
        fft = _fft_ratio(img)

        return {
            "label": final_label,
            "confidence": final_confidence,
            "p_real": p_real,
            "p_fake": p_fake,
            "ela": ela,
            "fft_ratio": fft,
        }

    def predict_image(self, image_input: Union[str, Path, Image.Image]):
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
            file_name = Path(image_input).name
        else:
            img = image_input.convert("RGB")
            file_name = "uploaded_image"

        result = self.predict_pil(img)
        result["file"] = file_name
        return result