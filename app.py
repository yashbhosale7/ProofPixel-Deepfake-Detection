import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import gradio as gr
from PIL import Image

from image_pipeline import DeepfakeImagePredictor
from video_pipeline import VideoDeepfakePredictor


image_predictor = DeepfakeImagePredictor()
video_predictor = VideoDeepfakePredictor(image_predictor=image_predictor)


# ---------------------------
# Helpers
# ---------------------------
def ensure_browser_playable_video(input_path: str) -> str:
    """
    Try to convert uploaded video to a browser-friendly MP4 (H.264 + AAC).
    If ffmpeg is unavailable or conversion fails, fall back to original file.
    """
    if input_path is None or not Path(input_path).exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    suffix = Path(input_path).suffix.lower()
    if suffix == ".mp4":
        # Still create a stable temp copy for preview/prediction
        temp_dir = tempfile.mkdtemp(prefix="proofpixel_video_")
        output_path = str(Path(temp_dir) / "preview.mp4")
        shutil.copy2(input_path, output_path)
        return output_path

    temp_dir = tempfile.mkdtemp(prefix="proofpixel_video_")
    output_path = str(Path(temp_dir) / "preview.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if Path(output_path).exists():
            return output_path
    except Exception:
        # fallback: copy original file if conversion fails
        fallback_path = str(Path(temp_dir) / Path(input_path).name)
        shutil.copy2(input_path, fallback_path)
        return fallback_path

    return input_path


def handle_video_upload(video_file):
    """
    Called immediately after upload.
    Returns:
      1. browser-playable preview path
      2. internal state path for prediction
      3. status message
    """
    try:
        if video_file is None:
            return None, None, "No video uploaded."

        raw_path = video_file if isinstance(video_file, str) else video_file
        playable_path = ensure_browser_playable_video(raw_path)

        return playable_path, playable_path, f"Video loaded successfully: {Path(playable_path).name}"

    except Exception as e:
        return None, None, f"Upload/preview error: {type(e).__name__}: {str(e)}"


def predict_image_ui(image: Image.Image):
    try:
        if image is None:
            return "No image uploaded.", "", ""

        result = image_predictor.predict_image(image)

        label_text = f"Prediction: {result['label']}"
        confidence_text = f"Confidence: {result['confidence'] * 100:.2f}%"
        details_text = (
            f"Real Probability: {result['p_real'] * 100:.2f}%\n"
            f"Fake Probability: {result['p_fake'] * 100:.2f}%\n"
            f"ELA Score: {result['ela']:.4f}\n"
            f"FFT Ratio: {result['fft_ratio']:.4f}"
        )

        return label_text, confidence_text, details_text

    except Exception as e:
        return "Error", "Error", f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"


def predict_video_ui(video_path):
    try:
        if video_path is None:
            return "No video uploaded.", "", ""

        result = video_predictor.predict_video(video_path)

        label_text = f"Prediction: {result['label']}"
        confidence_text = f"Confidence: {result['confidence'] * 100:.2f}%"
        details_text = (
            f"Real Probability: {result['p_real'] * 100:.2f}%\n"
            f"Fake Probability: {result['p_fake'] * 100:.2f}%\n"
            f"Frames Processed: {result['frames_processed']}\n"
            f"Faces Found: {result['faces_found']}\n"
            f"Frames Predicted Fake: {result['fake_frames']}\n"
            f"Frames Predicted Real: {result['real_frames']}\n"
            f"Details: {result['details']}"
        )

        return label_text, confidence_text, details_text

    except Exception as e:
        return "Error", "Error", f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="ProofPixel – Deepfake Detection System") as demo:
    gr.Markdown("# ProofPixel – Deepfake Detection System")
    gr.Markdown("Upload an image or video to predict whether it is Real or Fake.")

    with gr.Tabs():
        with gr.Tab("Image Detection"):
            img_input = gr.Image(type="pil", label="Upload Image")
            img_btn = gr.Button("Predict Image")
            img_label = gr.Textbox(label="Result")
            img_confidence = gr.Textbox(label="Confidence")
            img_details = gr.Textbox(label="Details", lines=8)

            img_btn.click(
                fn=predict_image_ui,
                inputs=img_input,
                outputs=[img_label, img_confidence, img_details],
            )

        with gr.Tab("Video Detection"):
            gr.Markdown("Upload a video. A browser-friendly preview will appear below.")

            video_input = gr.File(
                label="Upload Video File",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                type="filepath",
            )

            video_preview = gr.Video(label="Preview Video")
            video_state = gr.State()
            video_status = gr.Textbox(label="Video Status")

            video_input.change(
                fn=handle_video_upload,
                inputs=video_input,
                outputs=[video_preview, video_state, video_status],
            )

            video_btn = gr.Button("Predict Video")
            video_label = gr.Textbox(label="Result")
            video_confidence = gr.Textbox(label="Confidence")
            video_details = gr.Textbox(label="Details", lines=10)

            video_btn.click(
                fn=predict_video_ui,
                inputs=video_state,
                outputs=[video_label, video_confidence, video_details],
            )

demo.launch(share=True, inbrowser=True) 