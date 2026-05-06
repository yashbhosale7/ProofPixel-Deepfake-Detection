from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"
VIDEO_DATA_DIR = ROOT / "data_videos"

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
FALLBACK_CHECKPOINT = CHECKPOINT_DIR / "resnet50_best.pth"

MODEL_NAME = "resnet50"
NUM_CLASSES = 2
CLASS_NAMES = ["fake", "real"]

IMAGE_SIZE = 224
IMAGE_RESIZE = 256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VIDEO_SAMPLE_EVERY_N_FRAMES = 15
VIDEO_MAX_FRAMES = 20
VIDEO_FAKE_THRESHOLD = 0.5

HAAR_CASCADE_PATH = "/System/Library/Frameworks/Quartz.framework/Versions/A/Frameworks/ImageKit.framework/Versions/A/Resources/facehaar.xml"


def get_checkpoint_path() -> Path:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    if FALLBACK_CHECKPOINT.exists():
        return FALLBACK_CHECKPOINT
    raise FileNotFoundError(
        f"No checkpoint found. Checked: {DEFAULT_CHECKPOINT} and {FALLBACK_CHECKPOINT}"
    )