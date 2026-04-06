import os
import io
import zipfile
import tempfile
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "best_resnet50_finetuned.keras"
AUDIT_LOG_PATH = "audit_log.csv"
CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]
IMAGE_SIZE = (320, 320)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vehicle Damage Severity Detection",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def preprocess_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = image.copy()

    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return display_image, image_array


def predict_single(model, image_array):
    probs = model(image_array, training=False).numpy()[0]
    pred_index = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    return pred_class, confidence, probs


def generate_sha256(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def save_record(claim_id, pred_class, confidence, image_hash):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_id": claim_id,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "image_hash": image_hash
    }

    if os.path.exists(AUDIT_LOG_PATH):
        df = pd.read_csv(AUDIT_LOG_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(AUDIT_LOG_PATH, index=False)
    return record


def severity_explanation(pred_class):
    if pred_class == "01-minor":
        return "Low visible damage such as small dents or light scratches."
    elif pred_class == "02-moderate":
        return "Noticeable damage that may affect multiple body parts."
    elif pred_class == "03-severe":
        return "Heavy visible damage likely requiring major repair."
    return "Unknown severity."


def is_image_file(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

# =========================================================
# UI
# =========================================================
st.title("🚗 Vehicle Damage Severity Detection System")
st.write("Single-image prediction and batch evaluation using a fine-tuned ResNet50 model.")

st.sidebar.header("Project Information")
st.sidebar.write("**Model:** Fine-tuned ResNet50")
st.sidebar.write("**Input Size:** 320 × 320")
st.sidebar.write("**Classes:** Minor, Moderate, Severe")
st.sidebar.write("**Course:** Practical Deep Learning – Winter 2026")
st.sidebar.write("**Extension:** Blockchain-ready audit trail")

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

mode = st.radio(
    "Select Mode",
    ["Single Image Prediction", "Batch Prediction", "Batch Evaluation"],
    horizontal=True
)

# =========================================================
# 1) SINGLE IMAGE PREDICTION
# =========================================================
if mode == "Single Image Prediction":
    claim_id = st.text_input("Enter Claim ID", placeholder="Example: CLM-2026-001")
    uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        display_image, image_array = preprocess_image_bytes(image_bytes)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(display_image, caption=uploaded_file.name, width="stretch")

        with col2:
            st.subheader("Prediction")
            if st.button("Predict Damage Severity", key="predict_single"):
                progress_text = st.empty()
                progress_bar = st.progress(0)

                progress_text.text("Starting prediction... 0%")
                progress_bar.progress(20)

                progress_text.text("Preprocessing image... 20%")
                progress_bar.progress(50)

                pred_class, confidence, probs = predict_single(model, image_array)

                progress_text.text("Finalizing result... 80%")
                progress_bar.progress(100)

                st.success(f"Predicted Damage Severity: {pred_class}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.write(f"**Interpretation:** {severity_explanation(pred_class)}")

                prob_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability": probs
                })
                st.dataframe(prob_df, width="stretch")

                image_hash = generate_sha256(image_bytes)
                st.code(image_hash)

                if claim_id.strip():
                    saved_record = save_record(
                        claim_id=claim_id.strip(),
                        pred_class=pred_class,
                        confidence=confidence,
                        image_hash=image_hash
                    )
                    st.info("Prediction record saved.")
                    st.json(saved_record)

                progress_bar.empty()
                progress_text.empty()

# =========================================================
# 2) BATCH PREDICTION
# =========================================================
elif mode == "Batch Prediction":
    st.subheader("Batch Prediction")
    st.write("Upload multiple vehicle images. The app will predict severity for each image and produce a downloadable CSV.")

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_predict"
    )

    if uploaded_files:
        st.write(f"Total uploaded images: {len(uploaded_files)}")

        if st.button("Run Batch Prediction", key="run_batch_prediction"):
            results = []
            progress_bar = st.progress(0)
            progress_text = st.empty()

            total = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files, start=1):
                image_bytes = uploaded_file.getvalue()
                _, image_array = preprocess_image_bytes(image_bytes)

                pred_class, confidence, probs = predict_single(model, image_array)
                image_hash = generate_sha256(image_bytes)

                results.append({
                    "filename": uploaded_file.name,
                    "predicted_class": pred_class,
                    "confidence": round(confidence, 4),
                    "prob_01_minor": round(float(probs[0]), 4),
                    "prob_02_moderate": round(float(probs[1]), 4),
                    "prob_03_severe": round(float(probs[2]), 4),
                    "image_hash": image_hash
                })

                percent = int((i / total) * 100)
                progress_bar.progress(percent)
                progress_text.text(f"Processing image {i}/{total} ... {percent}%")

            results_df = pd.DataFrame(results)
            st.success("Batch prediction completed.")
            st.dataframe(results_df, width="stretch")

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

            progress_bar.empty()
            progress_text.empty()

# =========================================================
# 3) BATCH EVALUATION
# =========================================================
elif mode == "Batch Evaluation":
    st.subheader("Batch Evaluation")
    st.write(
        "Upload a ZIP file containing labeled class folders:\n"
        "`01-minor/`, `02-moderate/`, `03-severe/`.\n"
        "The app will predict all images and compute performance metrics."
    )

    zip_file = st.file_uploader("Upload labeled ZIP dataset", type=["zip"], key="batch_eval")

    if zip_file is not None:
        if st.button("Run Batch Evaluation", key="run_batch_evaluation"):
            results = []
            progress_bar = st.progress(0)
            progress_text = st.empty()

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getvalue())

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                image_paths = []
                true_labels = []

                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if is_image_file(file):
                            full_path = os.path.join(root, file)

                            # Label from parent folder
                            parent = os.path.basename(os.path.dirname(full_path))
                            if parent in CLASS_NAMES:
                                image_paths.append(full_path)
                                true_labels.append(parent)

                if len(image_paths) == 0:
                    st.error("No labeled images found. Make sure the ZIP contains class folders.")
                    st.stop()

                total = len(image_paths)

                for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels), start=1):
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()

                    _, image_array = preprocess_image_bytes(image_bytes)
                    pred_class, confidence, probs = predict_single(model, image_array)

                    results.append({
                        "filename": os.path.basename(img_path),
                        "true_label": true_label,
                        "predicted_label": pred_class,
                        "confidence": round(confidence, 4),
                        "correct": int(true_label == pred_class)
                    })

                    percent = int((i / total) * 100)
                    progress_bar.progress(percent)
                    progress_text.text(f"Evaluating image {i}/{total} ... {percent}%")

            results_df = pd.DataFrame(results)

            y_true = results_df["true_label"].tolist()
            y_pred = results_df["predicted_label"].tolist()

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
            clf_report = classification_report(y_true, y_pred, labels=CLASS_NAMES, output_dict=True)
            clf_report_df = pd.DataFrame(clf_report).transpose()

            st.success("Batch evaluation completed.")
            st.metric("Accuracy", f"{acc * 100:.2f}%")

            st.subheader("Evaluation Results")
            st.dataframe(results_df, width="stretch")

            st.subheader("Classification Report")
            st.dataframe(clf_report_df, width="stretch")

            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(cm, CLASS_NAMES)
            st.pyplot(fig)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Evaluation Results CSV",
                data=csv,
                file_name="batch_evaluation_results.csv",
                mime="text/csv"
            )

            progress_bar.empty()
            progress_text.empty()

# =========================================================
# AUDIT LOG VIEW
# =========================================================
st.markdown("---")
if st.checkbox("Show saved prediction records"):
    if os.path.exists(AUDIT_LOG_PATH):
        log_df = pd.read_csv(AUDIT_LOG_PATH)
        st.dataframe(log_df, width="stretch")
    else:
        st.info("No records saved yet.")