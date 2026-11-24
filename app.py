import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from kaggle_sync import CLASSES_PATH, ensure_model

IMG_SIZE = (224, 224)

CLASS_INFO = {
    "covid": ("COVID-19", "🦠", "#d62728"),
    "normal": ("Normal", "✅", "#2ca02c"),
    "pneumonia": ("Pneumonia", "🫁", "#ff7f0e"),
}

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="wide")


@st.cache_resource(show_spinner="Loading model (first run downloads it from Kaggle)...")
def load_artifacts():
    import tensorflow as tf

    model_path = ensure_model()
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(CLASSES_PATH.read_text())
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    # Training used RGB 224x224 images scaled to [0, 1] (Rescaling(1/255) sat outside the model).
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    return np.asarray(image, dtype="float32")[None] / 255.0


def predict(model, image: Image.Image) -> np.ndarray:
    return model.predict(preprocess(image), verbose=0)[0]


def label_of(name: str):
    return CLASS_INFO.get(name, (name.title(), "🩻", "#1f77b4"))


with st.sidebar:
    st.header("About")
    st.write(
        "A CNN trained on the *COVID19 / Pneumonia / Normal Chest X-ray (PA)* dataset "
        "classifies a chest X-ray into one of three classes."
    )
    st.markdown("**Input:** 224×224 RGB  \n**Output:** 3-class softmax  \n**Val. accuracy:** ≈ 89%")
    st.divider()
    show_chart = st.toggle("Show probability chart", value=True)
    threshold = st.slider("Confidence warning threshold", 0.5, 0.99, 0.70, 0.01)
    st.divider()
    st.warning(
        "Educational demo only. This is **not** a medical device and must not be used "
        "for diagnosis. Consult a qualified clinician."
    )

st.title("🫁 Chest X-ray Classifier")
st.caption("Upload a chest X-ray (PA view) to classify it as COVID-19, Pneumonia or Normal.")

try:
    model, class_names = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("1. Upload")
    source = st.radio("Input", ["Upload file", "Use camera"], horizontal=True, label_visibility="collapsed")
    if source == "Upload file":
        file = st.file_uploader("Chest X-ray image", type=["png", "jpg", "jpeg"])
    else:
        file = st.camera_input("Take a picture of an X-ray")

    image = None
    if file is not None:
        try:
            image = Image.open(file)
            image.load()
        except Exception:
            st.error("This file could not be read as an image.")
            image = None
    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

with right:
    st.subheader("2. Result")
    if image is None:
        st.info("Upload an image to see the prediction.")
    else:
        with st.spinner("Analyzing..."):
            probs = predict(model, image)
        top = int(np.argmax(probs))
        name, icon, color = label_of(class_names[top])
        conf = float(probs[top])

        st.markdown(
            f"<div style='padding:1rem 1.2rem;border-radius:12px;border-left:8px solid {color};"
            f"background:{color}22'><div style='font-size:0.9rem;opacity:.7'>Predicted class</div>"
            f"<div style='font-size:2rem;font-weight:700'>{icon} {name}</div>"
            f"<div style='font-size:1.1rem'>Confidence: <b>{conf:.1%}</b></div></div>",
            unsafe_allow_html=True,
        )
        if conf < threshold:
            st.warning(f"Low confidence (< {threshold:.0%}). Treat this result with extra caution.")

        st.markdown("&nbsp;")
        cols = st.columns(len(class_names))
        for col, cname, p in zip(cols, class_names, probs):
            col.metric(label_of(cname)[0], f"{p:.1%}")
        if show_chart:
            st.bar_chart({label_of(c)[0]: float(p) for c, p in zip(class_names, probs)}, horizontal=True)