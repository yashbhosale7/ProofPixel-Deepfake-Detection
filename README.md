![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Computer Vision](https://img.shields.io/badge/AI-ComputerVision-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)


## Repository Status

This repository contains the source code, inference pipeline, notebooks, and UI implementation for the ProofPixel Deepfake Detection System.

Due to GitHub storage limitations, datasets and trained model checkpoints are excluded from the repository.



# ProofPixel – Deepfake Detection System
An end-to-end AI system for detecting manipulated images and videos using Deep Learning and Computer Vision.


## Features

- Image Deepfake Detection
- Video Deepfake Detection
- Confidence Score Prediction
- Frame Extraction & Analysis
- Face Detection Pipeline
- Gradio-based Interactive UI
- Real/Fake Probability Breakdown
- CNN-based Image Classification
- CNN + LSTM-based Video Analysis

---

## Demo Capabilities

### Image Detection
- Upload manipulated or real facial images
- Predict Real vs Fake
- Confidence score generation
- Probability breakdown analysis

### Video Detection
- Upload video files
- Automatic frame extraction
- Face detection across sampled frames
- Aggregated sequence-based prediction

  

## Tech Stack

### AI / Deep Learning
- PyTorch
- ResNet50
- CNN + LSTM
- OpenCV
- NumPy
- Pandas
- Scikit-learn

### UI / Deployment
- Gradio

### Data Processing
- Albumentations
- Frame Extraction
- Image Preprocessing
- Video Sampling

---

## System Workflow

### Image Detection Pipeline
1. Upload Image
2. Preprocess Image
3. Run Deep Learning Inference
4. Generate Confidence Scores
5. Display Real/Fake Prediction

### Video Detection Pipeline
1. Upload Video
2. Extract Frames
3. Detect Faces
4. Process Sampled Frames
5. Run CNN + LSTM Inference
6. Aggregate Predictions
7. Display Final Prediction

---

## Screenshots

### Homepage
<img width="1407" height="867" alt="Screenshot 2026-05-06 at 17 00 49" src="https://github.com/user-attachments/assets/d632ef76-1023-4ec5-9600-b11938f23b17" />

### Image Detection
<img width="1424" height="897" alt="Screenshot 2026-05-06 at 17 01 16" src="https://github.com/user-attachments/assets/6e67ca8a-fccb-4d17-aebd-1e2bc8c68950" />

### Video Detection
<img width="1434" height="896" alt="Screenshot 2026-05-06 at 17 01 47" src="https://github.com/user-attachments/assets/bbaf8d9f-ef31-4348-b54a-8f3d4cc84892" />

---


## Project Structure

```bash
ProofPixel-Deepfake-Detection/
│
├── app.py
├── config.py
├── image_pipeline.py
├── video_pipeline.py
├── requirements.txt
│
├── notebooks/
├── scripts/
└── README.md
```




## Installation

Clone the repository:
```bash
git clone https://github.com/yashbhosale7/ProofPixel-Deepfake-Detection.git
cd ProofPixel-Deepfake-Detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```


Run the application:
```bash
python app.py
```


## Project Highlights

- Trained deepfake image classification models on 140K+ real and fake images.
- Implemented image and video deepfake detection pipelines using Deep Learning and Computer Vision.
- Built an interactive Gradio-based web interface for real-time inference.
- Added frame extraction and face analysis for video-based deepfake detection.
- Integrated confidence scoring and probability-based prediction outputs.




## Model Architecture

### Image Detection
- ResNet50-based CNN architecture
- Image preprocessing using Albumentations
- Binary classification for Real vs Fake prediction

### Video Detection
- CNN + LSTM pipeline
- Frame extraction and sampling
- Sequential frame-based prediction aggregation
- Face detection and temporal analysis





## Sample Prediction Output

### Image Prediction
- Prediction: Fake
- Confidence: 98.44%

### Video Prediction
- Prediction: Fake
- Confidence: 59.23%
- Frames Processed: 20



## Dataset

The project was trained using publicly available deepfake datasets containing real and manipulated facial images/videos.

- 140K+ training images
- Real and AI-generated face samples
- Frame-based video datasets for temporal analysis

Note: Raw datasets and trained checkpoints are excluded from the repository due to storage limitations.




## Current Limitations

- Video predictions depend on face visibility and frame quality.
- Performance may vary for heavily compressed videos.
- Real-time webcam inference is not yet implemented.
- Limited optimization for low-resource devices.






## Future Improvements

- Real-time webcam deepfake detection
- Transformer-based architectures (ViT / TimeSformer)
- Cloud deployment using Hugging Face Spaces
- REST API support for external integration
- Explainable AI heatmaps and visualizations
- Multi-face video analysis
- Mobile-friendly deployment




## Author

**Yash Bhosale**

- GitHub: https://github.com/yashbhosale7
- LinkedIn: https://www.linkedin.com/in/yash-bhosale-446593360/
