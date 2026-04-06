import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from itertools import cycle
from config import *

os.makedirs('outputs/mobilenetv2', exist_ok=True)

# ── Data generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.3, shear_range=0.2,
    horizontal_flip=True, brightness_range=[0.8, 1.2])

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_gen   = ImageDataGenerator(rescale=1./255).flow_from_directory(VAL_DIR,
                target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='categorical', shuffle=False)
test_gen  = ImageDataGenerator(rescale=1./255).flow_from_directory(TEST_DIR,
                target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='categorical', shuffle=False)

class_names = list(train_gen.class_indices.keys())

cw = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(cw))

# ── Model ─────────────────────────────────────────────────────────────────────
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

x      = GlobalAveragePooling2D()(base.output)
x      = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model  = Model(inputs=base.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ── Training ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('outputs/mobilenetv2/best_model.h5', save_best_only=True, verbose=1)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                    class_weight=class_weights, callbacks=callbacks)

# ── Plot & evaluate (same helpers as CNN script) ──────────────────────────────
def plot_history(history, save_path):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],     label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.show()

plot_history(history, 'outputs/mobilenetv2/training_curves.png')

def evaluate(generator, split_name, save_prefix):
    generator.reset()
    y_true       = generator.classes
    y_pred_probs = model.predict(generator, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    print(f"\n── {split_name} Classification Report ──")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {split_name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig(f'{save_prefix}_confusion_matrix.png'); plt.show()
    y_bin   = label_binarize(y_true, classes=[0,1,2])
    colors  = cycle(['darkorange','green','blue'])
    plt.figure(figsize=(8,6))
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_pred_probs[:,i])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--'); plt.grid()
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC Curves — {split_name}')
    plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(f'{save_prefix}_roc.png'); plt.show()

evaluate(val_gen,  'Validation', 'outputs/mobilenetv2/val')
evaluate(test_gen, 'Test',       'outputs/mobilenetv2/test')