import os
import time
import copy
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# 1. PATHS
# ============================================
TRAIN_DIR = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/training"
VAL_DIR   = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/validation"
TEST_DIR  = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/data3a/test"

PROJECT_DIR = "/Users/mdrabiulhasan/Documents/UofR/Deep-Learning/Project/carDamageSeverity"
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "fine_tuned_resnet50_vehicle_damage.pth")

# ============================================
# 2. DEVICE
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# 3. IMAGE TRANSFORMS
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================
# 4. LOAD DATASETS
# ============================================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=VAL_DIR, transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ============================================
# 5. LOAD PRETRAINED RESNET50
# ============================================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

# ============================================
# 6. LOSS AND OPTIMIZER
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ============================================
# 7. TRAINING SETTINGS
# ============================================
epochs = 10
history = {
    'train_accuracy': [],
    'val_accuracy': [],
    'train_loss': [],
    'val_loss': []
}

best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

start_time = time.time()

# ============================================
# 8. STAGE 1: TRAIN CLASSIFICATION HEAD
# ============================================
print("\nStarting Stage 1: Train classification head only\n")

for epoch in range(epochs):
    # ---------------- TRAIN ----------------
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = 100.0 * correct_train / total_train

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = 100.0 * correct_val / total_val

    history['train_accuracy'].append(epoch_train_acc)
    history['val_accuracy'].append(epoch_val_acc)
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Accuracy: {epoch_train_acc:.2f}%")
    print(f"Val Loss:   {epoch_val_loss:.4f} | Val Accuracy:   {epoch_val_acc:.2f}%")
    print("-" * 50)

    # Save best model weights by validation accuracy
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

end_time = time.time()
print(f"\nStage 1 training completed in {(end_time - start_time):.2f} seconds")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# Load best weights from Stage 1
model.load_state_dict(best_model_wts)

# ============================================
# 9. STAGE 2: FINE-TUNE TOP RESNET LAYERS
# ============================================
print("\nStarting Stage 2: Fine-tuning top ResNet50 layers\n")

# Unfreeze last few layers
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

# New optimizer for fine-tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

fine_tune_epochs = 10
best_model_wts_ft = copy.deepcopy(model.state_dict())
best_val_acc_ft = best_val_acc

ft_start_time = time.time()

for epoch in range(fine_tune_epochs):
    # ---------------- TRAIN ----------------
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = 100.0 * correct_train / total_train

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = 100.0 * correct_val / total_val

    print(f"[Fine-Tune] Epoch {epoch+1}/{fine_tune_epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Accuracy: {epoch_train_acc:.2f}%")
    print(f"Val Loss:   {epoch_val_loss:.4f} | Val Accuracy:   {epoch_val_acc:.2f}%")
    print("-" * 50)

    if epoch_val_acc > best_val_acc_ft:
        best_val_acc_ft = epoch_val_acc
        best_model_wts_ft = copy.deepcopy(model.state_dict())

ft_end_time = time.time()
print(f"\nFine-tuning completed in {(ft_end_time - ft_start_time):.2f} seconds")
print(f"Best Fine-Tuned Validation Accuracy: {best_val_acc_ft:.2f}%")

# Load best fine-tuned weights
model.load_state_dict(best_model_wts_ft)

# ============================================
# 10. TEST EVALUATION
# ============================================
print("\nEvaluating on test dataset...\n")

model.eval()
correct_test = 0
total_test = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total_test += labels.size(0)
        correct_test += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100.0 * correct_test / total_test
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ============================================
# 11. SAVE MODEL IN PROJECT FOLDER
# ============================================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nFine-tuned model saved successfully at:\n{MODEL_SAVE_PATH}")