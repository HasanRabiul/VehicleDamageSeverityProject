import os
import io
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from blockchain_client import load_config, InsuranceAuditClient

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
# SESSION STATE INIT
# =========================================================
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_class" not in st.session_state:
    st.session_state.pred_class = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "probs" not in st.session_state:
    st.session_state.probs = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "display_image" not in st.session_state:
    st.session_state.display_image = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "image_hash" not in st.session_state:
    st.session_state.image_hash = None

# =========================================================
# LOAD MODEL ONCE
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# =========================================================
# LOAD BLOCKCHAIN CLIENT ONCE
# =========================================================
@st.cache_resource
def load_blockchain_client():
    config = load_config()
    return InsuranceAuditClient(config)

# =========================================================
# PREPROCESS IMAGE (CACHED)
# =========================================================
@st.cache_data
def preprocess_image_cached(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = image.copy()

    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return display_image, image_array

# =========================================================
# PREDICTION
# =========================================================
def predict_damage(model, image_array):
    probs = model(image_array, training=False).numpy()[0]
    pred_index = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    return pred_class, confidence, probs

# =========================================================
# SHA-256 HASH
# =========================================================
def generate_sha256(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# =========================================================
# SAVE RECORD
# =========================================================
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

# =========================================================
# CLASS INTERPRETATION
# =========================================================
def severity_explanation(pred_class):
    if pred_class == "01-minor":
        return "Low visible damage such as small dents or light scratches."
    elif pred_class == "02-moderate":
        return "Noticeable damage that may affect multiple body parts."
    elif pred_class == "03-severe":
        return "Heavy visible damage likely requiring major repair."
    return "Unknown severity."

# =========================================================
# CLEAR STATE FOR NEW IMAGE
# =========================================================
def reset_prediction_state():
    st.session_state.prediction_done = False
    st.session_state.pred_class = None
    st.session_state.confidence = None
    st.session_state.probs = None
    st.session_state.image_bytes = None
    st.session_state.display_image = None
    st.session_state.current_file_name = None
    st.session_state.image_hash = None

# =========================================================
# MAIN UI
# =========================================================
st.title("🚗 Vehicle Damage Severity Detection System")
st.write("Upload a vehicle image and get an automated damage severity prediction using a fine-tuned ResNet50 model.")

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("**Model:** Fine-tuned ResNet50")
st.sidebar.write("**Input Size:** 320 × 320")
st.sidebar.write("**Classes:** Minor, Moderate, Severe")
st.sidebar.write("**Course:** Practical Deep Learning – Winter 2026")
st.sidebar.write("**Extension:** Blockchain-ready audit trail")

# Load model
try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load blockchain client
blockchain_enabled = True
try:
    bc_client = load_blockchain_client()
    st.sidebar.success("Blockchain client connected.")
except Exception as e:
    blockchain_enabled = False
    bc_client = None
    st.sidebar.warning(f"Blockchain not connected: {e}")

# Inputs
claim_id = st.text_input("Enter Claim ID", placeholder="Example: CLM-2026-001")
uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

# Handle uploaded image
if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        reset_prediction_state()
        st.session_state.current_file_name = uploaded_file.name

    image_bytes = uploaded_file.getvalue()
    display_image, image_array = preprocess_image_cached(image_bytes)

    st.session_state.image_bytes = image_bytes
    st.session_state.display_image = display_image

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(display_image, caption=uploaded_file.name, width="stretch")

    with col2:
        st.subheader("Prediction")

        if st.button("Predict Damage Severity"):
            progress_text = st.empty()
            progress_bar = st.progress(0)

            progress_text.text("Starting prediction... 0%")
            time.sleep(0.1)

            progress_bar.progress(20)
            progress_text.text("Preprocessing image... 20%")
            time.sleep(0.1)

            progress_bar.progress(50)
            progress_text.text("Running model inference... 50%")

            pred_class, confidence, probs = predict_damage(model, image_array)

            progress_bar.progress(80)
            progress_text.text("Finalizing prediction... 80%")
            time.sleep(0.1)

            image_hash = generate_sha256(image_bytes)

            st.session_state.prediction_done = True
            st.session_state.pred_class = pred_class
            st.session_state.confidence = confidence
            st.session_state.probs = probs
            st.session_state.image_hash = image_hash

            progress_bar.progress(100)
            progress_text.text("Prediction complete. 100%")
            time.sleep(0.2)

            progress_bar.empty()
            progress_text.empty()

# Show prediction results only after prediction is clicked
if st.session_state.prediction_done:
    st.markdown("---")
    st.subheader("Prediction Result")

    pred_class = st.session_state.pred_class
    confidence = st.session_state.confidence
    probs = st.session_state.probs
    image_hash = st.session_state.image_hash

    if pred_class == "01-minor":
        st.success(f"Predicted Damage Severity: {pred_class}")
    elif pred_class == "02-moderate":
        st.warning(f"Predicted Damage Severity: {pred_class}")
    else:
        st.error(f"Predicted Damage Severity: {pred_class}")

    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.write(f"**Interpretation:** {severity_explanation(pred_class)}")

    if st.checkbox("Show class probabilities"):
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        })
        st.dataframe(prob_df, width="stretch")

    if st.checkbox("Show image SHA-256 hash"):
        st.code(image_hash)

    if st.button("Save Prediction Record"):
        if not claim_id.strip():
            st.warning("Please enter a Claim ID before saving.")
        else:
            saved_record = save_record(
                claim_id=claim_id.strip(),
                pred_class=pred_class,
                confidence=confidence,
                image_hash=image_hash
            )
            st.success("Prediction record saved successfully.")
            st.json(saved_record)

    # =====================================================
    # BLOCKCHAIN ACTIONS
    # =====================================================
    st.markdown("---")
    st.subheader("Blockchain Actions")

    if not claim_id.strip():
        st.info("Enter a Claim ID to use blockchain functions.")
    elif not blockchain_enabled:
        st.warning("Blockchain client is not connected. Check your .env settings.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Store on Sepolia Blockchain"):
                try:
                    confidence_bps = int(round(confidence * 10000))
                    tx_hash, receipt = bc_client.submit_claim(
                        claim_id=claim_id.strip(),
                        image_hash=image_hash,
                        predicted_class=pred_class,
                        confidence_bps=confidence_bps
                    )
                    st.success("Claim stored on Sepolia successfully.")
                    st.write("**Transaction Hash:**")
                    st.code(tx_hash)
                    st.write(f"**Receipt Status:** {receipt.get('status')}")
                except Exception as e:
                    st.error(f"Blockchain submission failed: {e}")

        with col_b:
            if st.button("Verify Hash on Blockchain"):
                try:
                    verified = bc_client.verify_image_hash(
                        claim_id=claim_id.strip(),
                        image_hash=image_hash
                    )
                    if verified:
                        st.success("Blockchain hash verification successful.")
                    else:
                        st.error("Blockchain hash verification failed.")
                except Exception as e:
                    st.error(f"Blockchain verification failed: {e}")

        if st.button("Read Claim from Blockchain"):
            try:
                claim_data = bc_client.get_claim(claim_id.strip())
                st.success("Claim read from blockchain successfully.")
                st.write(claim_data)
            except Exception as e:
                st.error(f"Blockchain read failed: {e}")

# Optional audit log view
st.markdown("---")
if st.checkbox("Show saved prediction records"):
    if os.path.exists(AUDIT_LOG_PATH):
        log_df = pd.read_csv(AUDIT_LOG_PATH)
        st.dataframe(log_df, width="stretch")
    else:
        st.info("No records saved yet.")