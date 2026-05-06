# Deepfake Detection - Colab Training Guide

This guide explains how to train your model on Google Colab and use it locally for video detection.

## 1. Train on Colab
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload `scripts/Deepfake_Training.ipynb` to Colab.
3.  **Runtime Setup**:
    - Go to `Runtime` > `Change runtime type`.
    - Select **T4 GPU** (Hardware accelerator).
4.  **Data Setup**:
    - The notebook is set up to download a dataset from Kaggle.
    - Go to your Kaggle Account > Create New API Token -> download `kaggle.json`.
    - Run the notebook; when prompted, upload `kaggle.json`.
5.  **Run Training**:
    - Execute all cells.
    - The model will train for 10 epochs (adjustable).
    - At the end, it will download `best_model.pth`.

## 2. Deploy Locally
Once you have `best_model.pth`:

1.  **Move the file**:
    Place `best_model.pth` into your local `checkpoints` folder:
    ```
    deepfake-detection/checkpoints/best_model.pth
    ```

2.  **Install Requirements** (if not already done):
    ```bash
    pip install -r scripts/requirements.txt
    ```

3.  **Run the Tool**:
    ```bash
    python scripts/upload_ui.py
    ```

4.  **Detect Videos**:
    - Open `http://localhost:5001`.
    - Upload an **image** or a **video** (.mp4, .mov).
    - The tool will extract frames, detect faces, and give you a Real/Fake score!

## Troubleshooting
- **No Face Detected**: Ensure the video has visible faces. The tool ignores frames without faces.
- **Slow Processing**: Analyzing a long video takes time. Watch the terminal logs for progress.
