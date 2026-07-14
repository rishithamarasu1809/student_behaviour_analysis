# Deep Learning-Based Multi-Student Behaviour Analysis for Emotionally Intelligent Robot Teacher Systems

A real-time, edge-deployable computer vision pipeline that monitors multiple students in a classroom — detecting **sleeping** and **talking** behaviour — to serve as the perceptual backbone for emotionally intelligent robot teacher platforms (e.g. NAO, Pepper).

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
  - [Face Detection — YOLOv9](#face-detection--yolov9)
  - [Multi-Person Tracking — DeepSORT](#multi-person-tracking--deepsort)
  - [Sleeping Detection — SleepingModelV4](#sleeping-detection--sleepingmodelv4)
  - [Talking Detection — TalkingModel](#talking-detection--talkingmodel)
  - [Temporal Smoothing & Behaviour Hierarchy](#temporal-smoothing--behaviour-hierarchy)
- [Dataset](#dataset)
- [Results](#results)
- [Deployment](#deployment)
- [Comparison with Existing Methods](#comparison-with-existing-methods)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

Manually supervising student attention at scale is intractable, and prior deep-learning approaches to classroom behaviour recognition are typically too heavy for edge devices, temporally unstable, or reliant on a single modality (visual-only). This project proposes a **modular, dual-stream, identity-aware** pipeline that:

- Detects and persistently tracks each student's face across video frames.
- Classifies each tracked student as **Awake / Sleeping** and **Talking / Not Talking**.
- Smooths noisy frame-level predictions into stable, per-student behaviour labels.
- Runs in real time on a **Raspberry Pi 4** with a live web dashboard.

## Key Contributions

- **SleepingModelV4** — a dual-stream sleeping-detection architecture fusing MobileNetV2 visual features with a 7-D Eye Aspect Ratio (EAR) geometric feature vector, reaching **96.98% accuracy** and **100% recall** (zero false negatives).
- **EyeAttention** — a novel spatial attention module with an anatomical prior that biases feature pooling toward the eye region, improving robustness to partial occlusion.
- **MouthAttention-enhanced TalkingModel** — audio-free talking detection on cropped mouth regions, achieving an **F1-score of 0.82**.
- **YOLOv9 + DeepSORT** integration for face detection and persistent identity tracking, with **zero identity switches** across all test videos.
- A **behaviour-hierarchy temporal smoother** that removes noisy frame-wise label oscillations.
- Validated **edge deployment** on Raspberry Pi 4 with Flask-based MJPEG streaming and CSV logging at 1 Hz inference.

## System Architecture

```
Video / Webcam / PiCamera2 Input
            │
            ▼
   YOLOv9 Face Detection
            │
            ▼
      DeepSORT Tracking
            │
            ▼
     Crop Face per Track
       ┌────┴────┐
       ▼         ▼
 EAR Extractor  Image Transform
       └────┬────┘
            ▼
 SleepingModelV4 (dual-stream)
       ┌────┴────┐
   Sleeping     Awake
                  │
                  ▼
         Crop Mouth Region
                  │
                  ▼
            TalkingModel
                  │
                  ▼
     Temporal Smoother (per Track ID)
                  │
                  ▼
             Decision Layer
       ┌──────────┴──────────┐
       ▼                     ▼
 Annotated Video        Summary CSV
```

## Methodology

### Face Detection — YOLOv9

A YOLOv9 model (`YOLOv9face.pt`) fine-tuned on face imagery, leveraging Programmable Gradient Information (PGI) and GELAN to mitigate information bottlenecks in deep networks.

| Parameter | Value |
|---|---|
| Input resolution | 640×640 |
| Confidence threshold | 0.60 |
| Minimum face size | 48×48 px |
| Output | `[x1, y1, x2, y2]` + confidence |

### Multi-Person Tracking — DeepSORT

Combines a Kalman-filtered motion state with a deep appearance embedding, solving frame-to-frame assignment via the Hungarian algorithm over a weighted sum of Mahalanobis (motion) and cosine (appearance) distances. Achieved **zero identity switches** across 31 tracked students and 787 frames — critical for the temporal smoother to correctly accumulate per-student evidence.

### Sleeping Detection — SleepingModelV4

A dual-stream architecture over 224×224 face crops:

**1. CNN Visual Stream**
- MobileNetV2 backbone → `[B, 1280, 7, 7]` feature map.
- **EyeAttention module**: two 1×1 convolutions (1280 → 128 → 1) produce an attention logit map, element-wise multiplied by a fixed anatomical prior (row weights `[0.5, 2.0, 3.0, 3.0, 2.0, 0.5, 0.3]`) before softmax normalisation:

  ```
  f_vis = Σ(i,j) w(i,j) · F(i,j),   w = softmax(conv(F) × prior)
  ```

**2. EAR Geometric Stream**
- 7-D EAR feature vector from MediaPipe FaceMesh landmarks (left eye: `362,385,387,263,373,380`; right eye: `33,160,158,133,153,144`).
- Standard EAR formula: `EAR = (‖P₂−P₆‖ + ‖P₃−P₅‖) / (2‖P₁−P₄‖)`
- Feature vector: left EAR, right EAR, mean EAR, `|left − right|` EAR, normalised left/right EAR, and a binary landmark-detection flag. Normalised with `EAR_closed = 0.20`, `EAR_open = 0.28`; a calibrated fallback vector is used when landmark detection fails.
- Projected through FC layers `7 → 32 → 64` (ReLU).

**Fusion & Classification**
- Concatenate 1280-D visual + 64-D EAR → 1344-D.
- `Dropout(0.3) → Linear(1344, 256) → ReLU → Dropout(0.2) → Linear(256, 2) → Softmax` → `P(Awake)`, `P(Sleeping)`.

**Training (2 phases)**
- Phase 1 (5 epochs): freeze MobileNetV2 backbone; train EyeAttention, EAR projection, classifier (Adam, lr = 3×10⁻⁴).
- Phase 2 (≤25 epochs): full fine-tuning with cosine annealing, early stopping (patience = 7), class-weighted cross-entropy, gradient clipping (max norm 1.0).
- Custom **SimulateGlasses** augmentation (25% probability: frame-only, tinted, dark variants) for robustness to eye occlusion.

### Talking Detection — TalkingModel

- Mouth region cropped from MediaPipe outer-lip landmarks, padded, resized to 224×224 (crops < 16×16 px discarded).
- MobileNetV2 backbone → `MouthAttention` (two 1×1 convolutions, no anatomical prior, softmax over spatial positions).
- `Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.2) → Linear(256, 2) → Softmax` → `P(NotTalking)`, `P(Talking)`.
- Trained with Adam (lr = 3×10⁻⁴), cross-entropy loss, 20 epochs; augmentation: random resized cropping, horizontal flip, rotation (±10°), colour jitter.

Both models use ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Temporal Smoothing & Behaviour Hierarchy

- Maintains rolling per-track counters of sleeping/talking predictions over a recent frame window.
- A student is labelled **Sleeping** if the proportion of sleeping predictions in the window exceeds a threshold.
- **Hierarchy**: Sleeping is checked first — if sleeping, the talking model is skipped (saving edge compute); otherwise the student is assessed for talking. This enforces logical consistency and prevents contradictory labels.

## Dataset

| Dataset | Description |
|---|---|
| **Sleeping Detection** | Face crops from publicly available sources, labelled Awake / Sleeping, standardised to 224×224, EAR features pre-extracted via MediaPipe FaceMesh and cached. |
| **Talking Detection** | Mouth-region crops labelled Talking / NotTalking (PyTorch `ImageFolder` Train/Val split), extracted via MediaPipe outer-lip landmarks. |
| **Person-Level Test Videos** | 5 clips, 31.50 s total @ 25 fps (787 frames, 31 unique person tracks), manually annotated per person per segment. |

## Results

**Frame-Level Sleeping Detection** (397 validation samples)

| Metric | Value |
|---|---|
| Accuracy | 96.98% |
| Recall (Sleeping) | 100.00% |
| Precision (Sleeping) | 94.20% |
| F1-Score | 97.01% |
| Specificity | 94.06% |
| AUC | 0.9863 |
| False Negatives | 0 |

**Frame-Level Talking Detection** (583 validation samples)

| Metric | Value |
|---|---|
| Accuracy | 81.30% |
| Recall (Talking) | 86.11% |
| Precision (Talking) | 78.23% |
| F1-Score | 81.98% |
| Specificity | 76.61% |
| AUC | 0.8687 |

**Person-Level Pipeline Evaluation** (5 videos, 31 persons, 787 frames)

| Metric | Value |
|---|---|
| Overall Accuracy | 77.42% |
| Weighted F1-Score | 0.7676 |
| Precision | 0.8264 |
| Recall | 0.7549 |
| Sleeping Recall (Person-Level) | 100% |
| DeepSORT ID Switches | 0 |

## Deployment

- **Hardware**: Raspberry Pi 4 (4 GB RAM), CPU-only inference.
- **Software**: Python 3, `ultralytics`, `deep-sort-realtime`, `mediapipe`, `opencv`, `picamera2`, `flask`.
- **Inference rate**: 1 Hz — sufficient for sustained behavioural states (seconds–minutes).
- **Live dashboard**: Flask MJPEG stream (port 5000) with bounding boxes, track IDs, labels, and probabilities overlaid.
- **Logging**: per-frame and per-person CSV export via the `/export_csv` endpoint.

## Comparison with Existing Methods

| Method | Task | Performance |
|---|---|---|
| Yang et al. [1] | Behaviour | mAP 49% |
| Zhang et al. [3] | Behaviour | mF1 53.18% |
| Zhang & Li [4] | Behaviour | Low in crowded scenes |
| Gao & Hang [6] | Behaviour | ~95% (lab-only) |
| **Ours (Sleeping)** | Sleeping | **F1 = 97%, Recall = 100%** |
| **Ours (Talking)** | Talking | **F1 = 82%** |

Unlike prior work, this system uniquely combines multi-stream feature fusion (CNN + EAR), identity-preserving tracking-aware temporal smoothing, and validated Raspberry Pi deployment — with no compared method reporting 100% sleeping recall.

## Future Work

1. Facial emotion recognition for valence/arousal classification.
2. Audio-visual fusion (co-located microphone) for whispered-speech talking detection.
3. Transformer-based backbones (e.g., EfficientFormer) for better accuracy-latency trade-offs on ARM hardware.
4. Expanded behaviour taxonomy: hand-raising, phone use, note-taking.
5. Closed-loop integration with adaptive learning management platforms for real-time robot teaching adjustments.

## Authors

- **Vallabineni Joshitha** — `142401040@smail.iitpkd.ac.in`
- **Marasu Lavanya Lakshmi Rishitha** — `142401022@smail.iitpkd.ac.in`
- **Mentor:** Dr. M. Sabarimalai Manikandan — `msm@iitpkd.ac.in`

Department of Electrical Engineering, Indian Institute of Technology Palakkad, Kerala – 678623, India.

## Acknowledgements

The authors thank the Department of Electrical Engineering, IIT Palakkad, and the OELP Coordinators for their continuous guidance and support throughout this research project.

## References

Selected references — see the full paper for the complete list:

1. X. Yang et al., "Improved YOLOv7 with spatio-temporal attention and Wise-IoU for classroom behaviour detection," *arXiv:2310.02523*, 2023.
2. H. Fang et al., "Real-time classroom behavior and drawing pattern analysis using YOLOv11," *Discover Computing*, vol. 29, 2026.
3. H. Zhang et al., "MSTA-SlowFast: Multi-scale spatial-temporal attention for student behaviour recognition," *Proc. IEEE ICME*, 2023.
4. T. Soukupova and J. Cech, "Real-time eye blink detection using facial landmarks," *Proc. 21st CVWW*, 2016.
5. C.-Y. Wang et al., "YOLOv9: Learning what you want to learn using programmable gradient information," *arXiv:2402.13616*, 2024.
6. N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," *Proc. IEEE ICIP*, 2017, pp. 3645–3649.
7. M. Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," *Proc. IEEE CVPR*, 2018, pp. 4510–4520.
