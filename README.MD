# CNN-Based Vehicle Damage Severity Detection Using Transfer Learning with ResNet50

This project develops a deep learning system for **vehicle damage severity classification** from images.  
The model predicts one of three classes:

- `01-minor`
- `02-moderate`
- `03-severe`

The main objective is to support **insurance claim triage** by providing a fast first-pass estimate of visible vehicle damage severity from uploaded images.

---

## Project Overview

Insurance claim handling increasingly starts with customer-submitted photographs instead of full in-person inspection. This project applies **computer vision** and **deep learning** to classify vehicle damage severity automatically.

A **transfer learning** approach with **ResNet50** is used as the main model. The project also includes comparison experiments with:

- Custom CNN
- MobileNetV2
- ResNet50
- YOLOv8

The final selected model for the main study is **ResNet50**, trained in two stages:

1. **Stage 1:** frozen-backbone feature extraction  
2. **Stage 2:** fine-tuning of upper layers

---

## Dataset

This project uses the **Car Damage Severity Dataset** from Kaggle.

**Dataset source:**  
[Car Damage Severity Dataset - Kaggle](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)

### Dataset splits used in this project
- **Training:** 1074 images
- **Validation:** 248 images
- **Test:** 277 images

### Class labels
- `01-minor`
- `02-moderate`
- `03-severe`

---

## Main Model Architecture

The selected final model uses **ResNet50** as a pre-trained convolutional backbone.

### Architecture summary
- Input image: `320 × 320 × 3`
- ResNet50 backbone (`include_top=False`)
- Global Average Pooling
- Dense layer: 256 units + ReLU
- Dropout: 0.4
- Dense layer: 128 units + ReLU
- Dropout: 0.3
- Softmax output: 3 classes

---

## Training Strategy

The model is trained in **two stages**:

### Stage 1: Frozen-backbone training
- ResNet50 backbone frozen
- Only classification head is trained

### Stage 2: Fine-tuning
- Upper backbone layers unfrozen
- Lower learning rate used
- Model adapts to vehicle-damage-specific patterns

This staged transfer-learning approach helps improve performance on a moderate-sized dataset.

---

## Project Structure

Example structure:

```text
deep-learning-project/
├── app.py
├── blockchain_client.py
├── fine_tuning.py
├── prediction_single_image.py
├── requirements.txt
├── README.md
├── model/
│   └── best_resnet50_finetuned.keras
├── figures/
├── report/
│   └── final_report.pdf
├── presentation/
│   └── final_presentation.pptx
└── outputs/
