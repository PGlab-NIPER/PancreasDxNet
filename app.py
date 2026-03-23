import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from utils.model import predict_image

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Pancreatic Disease Classifier", layout="centered")

st.title("Pancreatic Disease Classification (ResNet50)")
st.write("Upload a CT scan image to classify:")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")
st.sidebar.info("""
Model: ResNet50
Classes: Cancer, Normal, Pancreatitis
Input: Abdomen CT scan images
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # ✅ FIXED deprecated warning
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)

    if st.button("Predict"):
        try:
            label, probs = predict_image(image_np)

            # ---------------- RESULT ----------------
            st.subheader(f"Prediction: {label}")

            # ✅ Metric display
            st.metric("Confidence", f"{max(probs):.2f}")

            # ---------------- PROBABILITY CHART ----------------
            prob_df = pd.DataFrame({
                "Class": ["Cancer", "Normal", "Pancreatitis"],
                "Probability": probs
            })

            st.bar_chart(prob_df.set_index("Class"))

            # ---------------- RAW VALUES ----------------
            with st.expander("See raw probabilities"):
                st.write({
                    "Cancer": float(probs[0]),
                    "Normal": float(probs[1]),
                    "Pancreatitis": float(probs[2])
                })

            # ---------------- VISUAL ALERT ----------------
            if label == "Cancer":
                st.error("⚠️ Cancer detected")
            elif label == "Pancreatitis":
                st.warning("⚠️ Pancreatitis detected")
            else:
                st.success("✅ Normal")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
