import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Path to the .keras file produced by train_model.ipynb
# (Section 5: model.save('landslide_model.keras')).
# Keep the model file in the same folder as this app.py, or change this path.
MODEL_PATH = "landslide_model.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224

# IMPORTANT: tf.keras.utils.image_dataset_from_directory assigns class
# indices ALPHABETICALLY by subfolder name. Open train_model.ipynb, run the
# "Load the Dataset" cell, and copy the exact list printed as
# "Classes found: [...]" here, in that exact order. Getting this wrong will
# silently flip every prediction.
CLASS_NAMES = ["landslide", "no_landslide"]  # <-- update against the training output

st.set_page_config(
    page_title="Landslide Forecasting",
    page_icon="🛰️",
    layout="wide",
)


# --------------------------------------------------
# Model only loads once per session, not on every rerun
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Preprocessing + inference
# --------------------------------------------------

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resize/format an uploaded image to match what the model expects.

    NOTE: train_model.ipynb bakes `mobilenet_v2.preprocess_input` in as a
    layer INSIDE the model itself (Section 3), and the augmentation layers
    (RandomFlip/RandomRotation) are inactive at inference time. So here we
    only resize and hand over raw 0-255 pixel values -- do NOT divide by
    255 or normalise again, or predictions will be wrong.
    """
    img = pil_image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = tf.keras.utils.img_to_array(img)  # (224, 224, 3), float32, range 0-255
    arr = np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)
    return arr


def predict(pil_image: Image.Image):
    """Returns (label, confidence, raw_sigmoid_probability)."""
    arr = preprocess_image(pil_image)
    raw_prob = float(model.predict(arr, verbose=0)[0][0])  # P(class == CLASS_NAMES[1])

    if raw_prob >= 0.5:
        label, confidence = CLASS_NAMES[1], raw_prob
    else:
        label, confidence = CLASS_NAMES[0], 1 - raw_prob

    return label, confidence, raw_prob


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛰️ Landslide Forecasting")
st.write("Vision-Based Landslide Detection Using Satellite Imagery")
st.divider()

if model is None:
    st.warning(
        f"⚠️ Model file `{MODEL_PATH}` was not found next to `app.py`. "
        "Run `train_model.ipynb` to train and save the model, then copy "
        "the resulting `landslide_model.keras` into this app's folder "
        "(or update `MODEL_PATH`). The UI below still works for browsing, "
        "but prediction is disabled until the model file is present."
    )


# --------------------------------------------------
# Main sections
# --------------------------------------------------

upload_col, info_col = st.columns([2, 1])

with upload_col:
    st.subheader("Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG satellite image.",
    )

with info_col:
    st.subheader("About")
    st.write(
        "This system analyzes satellite imagery and predicts "
        "whether the input image is associated with a landslide."
    )
    st.caption(
        "Experimental academic prototype — not a certified or operational "
        "disaster-warning system."
    )

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    st.subheader("Image Preview")
    st.image(pil_image, caption=uploaded_file.name, use_container_width=True)

    predict_clicked = st.button(
        "🔍 Predict Landslide",
        type="primary",
        use_container_width=True,
        disabled=model is None,
    )

    if predict_clicked and model is not None:
        with st.spinner("Running model inference..."):
            label, confidence, raw_prob = predict(pil_image)

        st.subheader("Prediction Result")

        result_col, gauge_col = st.columns(2)

        with result_col:
            if label.lower() == "landslide":
                st.error(f"⚠️ **{label.upper()}** detected")
            else:
                st.success(f"✅ **{label.replace('_', ' ').upper()}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        with gauge_col:
            st.write("Raw model output (sigmoid probability):")
            st.progress(raw_prob)
            st.caption(f"P(class = \"{CLASS_NAMES[1]}\") = {raw_prob:.4f}")

        st.caption(
            "This prediction comes from an experimental transfer-learning model "
            "trained on limited satellite imagery. It should not be used as the "
            "sole basis for safety, evacuation, or operational decisions."
        )

st.divider()
st.caption(
    "Landslide boundary data for the Ditwah Cyclonic Storm (2025) provided by the "
    "Arthur C. Clarke Institute for Modern Technologies (ACCIMT). Used for academic, "
    "research, and decision-support purposes only — not verified for operational or "
    "legal use."
)