import numpy as np
import cv2
import os
import streamlit as st
from tensorflow.keras.models import load_model

# Class labels
class_labels = ['Cancer', 'Normal', 'Pancreatitis']

# Robust model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "generated_model.h5")

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

def preprocess_image_from_array(image, target_size=(240, 240)):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, target_size)

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)

    img = img / 255.0

    img_rgb = np.stack((img,) * 3, axis=-1)

    return img_rgb

def predict_image(image):
    if image is None:
        raise ValueError("Invalid image input")

    processed = preprocess_image_from_array(image)
    processed = np.expand_dims(processed, axis=0)

    preds = model.predict(processed, verbose=0)
    probs = preds[0]

    predicted_index = np.argmax(probs)
    predicted_label = class_labels[predicted_index]

    return predicted_label, probs
