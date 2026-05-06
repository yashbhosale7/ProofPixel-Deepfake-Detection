📄 ProofPixel – Deepfake Detection System

(Complete Technical Documentation)

1. Introduction

Deepfakes are AI-generated media where a person’s face, voice, or actions are synthetically manipulated using deep learning techniques. With the rapid advancement of generative models such as GANs (Generative Adversarial Networks) and diffusion models, distinguishing real content from manipulated content has become increasingly difficult.

This project, ProofPixel, is designed to detect deepfake images and videos using a hybrid approach combining:

Deep Learning (CNN – ResNet50)

Image Forensics (ELA + FFT)

Video Frame Analysis

The system provides a user-friendly interface where users can upload images or videos and receive a prediction indicating whether the content is Real or Fake.

2. Problem Statement

The rise of deepfake technology poses serious threats:

Misinformation and fake news

Identity theft and impersonation

Cybersecurity risks

Loss of trust in digital media

The problem is:

How can we automatically detect whether an image or video is real or manipulated using AI techniques?

3. Objectives

The main objectives of this project are:

Build a deep learning model to classify images as real or fake

Extend detection to videos using frame-based analysis

Integrate forensic features like ELA and FFT

Create an interactive UI for real-time detection

Achieve reasonable accuracy without heavy computation


4. System Overview

The system has two main pipelines:

4.1 Image Detection Pipeline:
Input Image → Preprocessing → CNN Model → Probabilities → Output
4.2 Video Detection Pipeline:
Input Video → Frame Sampling → Face Detection → Image Model → Aggregation → Output


5. Technologies Used
Category			Technology
Deep Learning		PyTorch
Model Architecture	ResNet50
Computer Vision		OpenCV
Image Processing	PIL, NumPy
UI Framework		Gradio
Model Library	    timm
Backend	            Python




6. Deep Learning Model
6.1 Model Used: ResNet50

ResNet50 is a convolutional neural network with 50 layers that uses skip connections (residual learning) to solve vanishing gradient problems.

Why ResNet50?

Proven performance in image classification

Deep architecture (captures complex patterns)

Efficient training and inference


6.2 Model Output

The model outputs:

[p_fake, p_real]

Final prediction:

If p_real ≥ p_fake → Real
Else → Fake





7. Image Processing Pipeline
7.1 Preprocessing

Each image undergoes:

Resize → 256

Center Crop → 224×224

Normalization (ImageNet stats)

7.2 Prediction Flow
Image → Tensor → Model → Softmax → Probabilities



7.3 Additional Forensic Features
7.3.1 Error Level Analysis (ELA)

ELA detects compression inconsistencies.

Logic:

Save image at lower JPEG quality

Compare original vs compressed

Measure pixel differences

Fake images often show uneven compression artifacts.

7.3.2 FFT (Frequency Analysis)

FFT converts image into frequency domain.

Logic:

Apply 2D Fourier Transform

Analyze high-frequency components

Deepfakes often have:

abnormal frequency patterns

unnatural smoothness or noise



8. Video Processing Pipeline
8.1 Frame Sampling

Instead of processing the entire video:

Sample every N frames (e.g., 15)
Max frames: 20

Reason:

reduces computation

speeds up inference

8.2 Face Detection

We use Haar Cascade (OpenCV).

Steps:

Convert frame to grayscale

Detect faces

Select largest face

Crop and resize




8.3 Frame-Level Prediction

Each detected face is passed through the image model:

Frame → Face Crop → CNN → Prediction
8.4 Aggregation Logic

We compute:

Average probability:

avg_p_fake = mean(p_fake across frames)
avg_p_real = mean(p_real across frames)

Final decision:

If avg_p_fake ≥ threshold → Fake
Else → Real









9. User Interface (Gradio)

The UI has two tabs:

Image Detection

Upload image

Predict

Shows:

Label

Confidence

ELA

FFT

Video Detection

Upload video

Preview video

Predict

Shows:

Label

Confidence

Frames processed

Faces detected

10. Key Features

Real-time prediction

Supports both image and video

Lightweight model (runs locally)

No retraining required

Hybrid approach (CNN + Forensics)

11. Limitations

Works best on face-based content

Haar cascade may miss small faces

Not 100% accurate (deepfakes are evolving)

Depends on training dataset quality

Non-face videos may give uncertain results


12. Future Improvements

Use better face detectors:

MTCNN

RetinaFace

Add Grad-CAM visualization

Use video models:

LSTM

3D CNN

Deploy on cloud

Improve dataset diversity

13. Applications

Social media verification

News authentication

Cybersecurity

Digital forensics

Law enforcement


14. Conclusion

ProofPixel demonstrates a practical approach to deepfake detection using a combination of deep learning and image forensics.

While not perfect, the system effectively identifies manipulated media in many real-world scenarios and serves as a strong foundation for further research and development.

15. Sample Output Explanation
Image Example:
Prediction: Real
Confidence: 97%
ELA: 0.69
FFT: 0.96

Video Example:
Prediction: Fake
Confidence: 59%
Frames processed: 20
Faces detected: 1

16. Architecture Diagram (Text)
                User Input
              (Image / Video)
                     │
         ┌───────────▼───────────┐
         │   Preprocessing Layer  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   CNN Model (ResNet50) │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ Forensic Features      │
         │  - ELA                │
         │  - FFT                │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │ Prediction   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Gradio UI  │
              └─────────────┘


17. Folder Structure

deepfake-detection/
│
├── checkpoints/
│   └── best.pt
│
├── data/
├── data_videos/
│
├── scripts/
│   └── infer.py
│
├── config.py
├── image_pipeline.py
├── video_pipeline.py
├── app.py
│
└── requirements.txt


18. Final Output

The final system allows:

✔ Upload image → detect real/fake
✔ Upload video → analyze frames → detect deepfake
✔ View probabilities and forensic metrics
✔ Interactive UI