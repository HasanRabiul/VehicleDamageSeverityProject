import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1) Paths (EDIT if needed)
# ----------------------------
TRAIN_DIR = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/training"
VAL_DIR   = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/validation"
TEST_DIR  = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/test"

# ----------------------------
# 2) Parameters
# ----------------------------
image_size = (224, 224)
batch_size = 32
num_classes = 3
epochs = 25
seed = 42

# ----------------------------
# 3) Data Generators
#   IMPORTANT: use preprocess_input (not rescale=1/255)
# ----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=seed
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# This MUST be categorical if you want evaluate() to work
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# ----------------------------
# 4) Build ResNet50 model
# ----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze CNN backbone initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 5) Train (NO steps_per_epoch / validation_steps)
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
]

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks
)

# ----------------------------
# 6) Evaluate on test (works now)
# ----------------------------
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------------
# 7) Predictions + report
# ----------------------------
pred_probs = model.predict(test_generator, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)

true_classes = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\nConfusion Matrix:\n", confusion_matrix(true_classes, pred_classes))
print("\nClassification Report:\n", classification_report(true_classes, pred_classes, target_names=class_names, digits=4))

# Print a few predictions
filenames = test_generator.filenames
for i in range(min(10, len(filenames))):
    print(f"{filenames[i]}  -->  Predicted: {class_names[pred_classes[i]]}")
