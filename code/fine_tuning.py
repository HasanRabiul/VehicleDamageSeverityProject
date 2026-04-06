import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# ----------------------------
# 1) Paths
# ----------------------------
TRAIN_DIR = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/training"
VAL_DIR   = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/validation"
TEST_DIR  = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/test"

# ----------------------------
# 2) Parameters
# ----------------------------
image_size = (320, 320)
batch_size = 16
num_classes = 3
initial_epochs = 15
fine_tune_epochs = 20
seed = 42

# ----------------------------
# 3) Data generators
# ----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=seed
)

validation_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# ----------------------------
# 4) Build model
# ----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(320, 320, 3)
)

# Stage 1: freeze full backbone
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

# ----------------------------
# 5) Compile for stage 1
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 6) Callbacks
# ----------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        verbose=1
    ),
    ModelCheckpoint(
        "best_resnet50_stage1.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ----------------------------
# 7) Stage 1 training
# ----------------------------
print("\nStarting Stage 1: Train classification head only\n")

history_stage1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=initial_epochs,
    callbacks=callbacks
)

# ----------------------------
# 8) Stage 2: fine-tune top layers
# ----------------------------
print("\nStarting Stage 2: Fine-tune top ResNet50 layers\n")

base_model.trainable = True

# Freeze lower layers, unfreeze top layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_finetune = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        verbose=1
    ),
    ModelCheckpoint(
        "best_resnet50_finetuned.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history_stage2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=fine_tune_epochs,
    callbacks=callbacks_finetune
)

# ----------------------------
# 9) Final evaluation
# ----------------------------
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------------
# Plot training and validation accuracy/loss
# ----------------------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_stage1.history['accuracy'], label='Train Accuracy (Stage 1)')
plt.plot(history_stage1.history['val_accuracy'], label='Val Accuracy (Stage 1)')

if 'accuracy' in history_stage2.history:
    offset = len(history_stage1.history['accuracy'])
    plt.plot(range(offset, offset + len(history_stage2.history['accuracy'])),
             history_stage2.history['accuracy'],
             label='Train Accuracy (Stage 2)')
    plt.plot(range(offset, offset + len(history_stage2.history['val_accuracy'])),
             history_stage2.history['val_accuracy'],
             label='Val Accuracy (Stage 2)')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_stage1.history['loss'], label='Train Loss (Stage 1)')
plt.plot(history_stage1.history['val_loss'], label='Val Loss (Stage 1)')

if 'loss' in history_stage2.history:
    offset = len(history_stage1.history['loss'])
    plt.plot(range(offset, offset + len(history_stage2.history['loss'])),
             history_stage2.history['loss'],
             label='Train Loss (Stage 2)')
    plt.plot(range(offset, offset + len(history_stage2.history['val_loss'])),
             history_stage2.history['val_loss'],
             label='Val Loss (Stage 2)')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_loss_curves.png", dpi=300)
plt.show()


# ----------------------------
# 10) Predictions and reports
# ----------------------------
pred_probs = model.predict(test_generator, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)

true_classes = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\nConfusion Matrix:\n")
print(confusion_matrix(true_classes, pred_classes))

print("\nClassification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=class_names, digits=4))

# Show a few sample predictions
filenames = test_generator.filenames
print("\nSample Predictions:")
for i in range(min(10, len(filenames))):
    print(f"{filenames[i]}  -->  Predicted: {class_names[pred_classes[i]]}")

# ----------------------------
# Confusion Matrix Heatmap
# ----------------------------
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ----------------------------
# ROC Curve for each class
# ----------------------------
true_binarized = label_binarize(true_classes, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(true_binarized[:, i], pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.show()

# ----------------------------
# Precision-Recall Curve for each class
# ----------------------------
plt.figure(figsize=(8, 6))

for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(true_binarized[:, i], pred_probs[:, i])
    ap_score = average_precision_score(true_binarized[:, i], pred_probs[:, i])
    plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap_score:.2f})")

plt.title("Precision-Recall Curve (One-vs-Rest)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curves.png", dpi=300)
plt.show()