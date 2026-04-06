import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================
# Paths
# =========================
MODEL_PATH = "best_resnet50_finetuned.keras"
IMAGE_PATH = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/carDamageSeverity/images/minor-1.jpeg"

# =========================
# Class names
# Must match training order
# =========================
class_names = ["01-minor", "02-moderate", "03-severe"]

# =========================
# Check files
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

# =========================
# Load model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# =========================
# Load and preprocess image
# IMPORTANT: use same size as training
# =========================
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((320, 320))

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

# =========================
# Predict
# =========================
pred_probs = model.predict(image_array, verbose=0)[0]
pred_index = int(np.argmax(pred_probs))
pred_class = class_names[pred_index]
confidence = float(pred_probs[pred_index])

# =========================
# Output
# =========================
print(f"\nPredicted class: {pred_class}")
print(f"Confidence: {confidence * 100:.2f}%")

print("\nAll class probabilities:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {pred_probs[i] * 100:.2f}%")