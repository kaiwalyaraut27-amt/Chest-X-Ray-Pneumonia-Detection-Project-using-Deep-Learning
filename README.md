# 🩺 Chest X-Ray Pneumonia Detection using Deep Learning

## Overview

This project is an end-to-end Deep Learning pipeline for detecting **Pneumonia from Chest X-Ray images** using **Transfer Learning with MobileNetV2** and TensorFlow/Keras.

The project covers the complete Deep Learning workflow:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Data Augmentation
* CNN Architecture Design
* Transfer Learning
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Model Comparison
* Model Saving & Metadata Management
* Production-Style Prediction Interface
* Project Documentation

The goal is to automatically classify a chest X-ray as:

* **NORMAL**
* **PNEUMONIA**

---

## Problem Statement

Pneumonia is a serious respiratory infection that affects millions of people worldwide. Early diagnosis is important because delayed treatment can lead to severe complications and increased mortality.

Radiologists typically diagnose pneumonia by examining chest X-rays. This project explores how Deep Learning can assist medical professionals by automatically identifying pneumonia patterns in X-ray images.

This problem is formulated as a **Binary Image Classification** task.

---

## Dataset

Dataset:

**Chest X-Ray Images (Pneumonia)**

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset Structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Dataset Statistics:

| Split      | Images |
| ---------- | -----: |
| Train      |   5230 |
| Validation |     16 |
| Test       |    624 |

Task Type:

* Binary Classification

Classes:

* NORMAL
* PNEUMONIA

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab

---

## Project Pipeline

### Stage 1 — Project Brief

* Problem definition
* Metric selection
* Architecture selection
* Transfer learning justification

### Stage 2 — Exploratory Data Analysis

* Dataset size inspection
* Class distribution analysis
* Sample visualization
* Image quality inspection

Outputs:

```text
eda_samples.png
eda_class_dist.png
```

---

### Stage 3 — Data Preprocessing

Performed:

* Image resizing (150×150)
* Pixel normalization (0–1)
* Data augmentation
* Batch preparation

Augmentations:

* Random Flip
* Random Rotation
* Random Zoom

Output:

```text
preprocess_augmented.png
```

---

### Stage 4 — Architecture Design

Base Model:

```text
MobileNetV2 (ImageNet Pretrained)
```

Classification Head:

```text
GlobalAveragePooling2D
BatchNormalization
Dropout
Dense(ReLU)
Dense(Sigmoid)
```

Transfer Learning was used to leverage pretrained visual features.

---

### Stage 5 — Model Training

Optimizer:

```text
Adam
```

Learning Rate:

```text
0.0001
```

Callbacks:

* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau
* CSVLogger

Output:

```text
training_curves.png
```

---

### Stage 6 — Evaluation

Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC
* Sensitivity
* Specificity

Outputs:

```text
confusion_matrix.png
roc_curve.png
```

---

### Stage 7 — Hyperparameter Tuning

Experiments Included:

* Baseline Model
* Dropout Adjustment
* Additional Dense Layer
* Fine-Tuning Top Layers

Goal:

Improve validation accuracy and reduce validation loss while minimizing overfitting.

---

### Stage 8 — Model Comparison

Compared:

* Baseline Model
* Tuned Models
* Transfer Learning Variants

Outputs:

```text
model_comparison.png
roc_comparison.png
```

---

### Stage 9 — Model Persistence

Saved Files:

```text
final_model.keras
best_model.keras
model_metadata.json
class_labels.json
training_log.csv
```

Metadata includes:

* Architecture information
* Input shape
* Class labels
* Metrics
* Training information

---

### Stage 10 — Prediction Interface

Implemented:

```python
DeepLearningPredictor
```

Features:

* Model loading
* Automatic preprocessing
* Prediction confidence
* Error handling
* Batch prediction

Output:

```text
prediction_grid.png
```

---

### Stage 11 — Documentation

Comprehensive project summary including:

* Problem statement
* Dataset analysis
* Architecture decisions
* Results
* Limitations
* Future work

---

## Model Performance

Replace with your final results:

| Metric      | Score   |
| ----------- | ------- |
| Accuracy    | 86.06X%  |
| Precision   | 83.01%  |
| Recall      | 97.69%  |
| F1 Score    | 89.75%  |
| AUC         | 0.9558 |
| Sensitivity | 97.69%  |
| Specificity | 66.67%  |

---

## Repository Structure

```text
.
├── dl_day6_final.py
├── PROJECT_SUMMARY.md
│
├── final_model.keras
├── best_model.keras
│
├── model_metadata.json
├── class_labels.json
├── training_log.csv
│
├── eda_samples.png
├── eda_class_dist.png
├── preprocess_augmented.png
├── training_curves.png
├── confusion_matrix.png
├── roc_curve.png
├── model_comparison.png
├── roc_comparison.png
├── prediction_grid.png
│
└── README.md
```

---

## Future Improvements

Potential improvements include:

* EfficientNetB0 / EfficientNetV2
* Grad-CAM Explainability
* Larger Validation Set
* Class Weighting
* Cross Validation
* Ensemble Models
* Deployment via FastAPI
* Docker Containerization

---

## Learning Outcomes

Through this project I learned:

* End-to-end Deep Learning workflows
* Transfer Learning techniques
* CNN architecture design
* Medical image classification
* Model evaluation beyond accuracy
* Hyperparameter tuning
* Model deployment fundamentals
* Production-style prediction pipelines

---

## Author

**Kaiwalya Raut**

Deep Learning Bootcamp — Day 6 Capstone Project

Built using TensorFlow, Keras, and Google Colab.
