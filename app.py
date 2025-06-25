import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Check if model file exists
model_path = "gender_model.h5"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the repository.")
    st.stop()

# Load model
try:
    model = load_model(model_path)
    st.write("Model loaded successfully")
    st.write(f"Expected input shape: {model.input_shape}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load and prep image
def load_and_prep(image, img_shape=224):
    try:
        image = image.resize((img_shape, img_shape))
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[..., :3]  # Remove alpha channel if present
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit UI
st.title("Gender Classifier")
st.write("Upload a face image (JPEG/PNG) to predict gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)
        image = load_and_prep(img)
        if image is not None:
            # Predict
            image_array = np.expand_dims(image, axis=0)
            pred = model.predict(image_array)
            pred_class = "Male" if tf.round(pred)[0][0] == 1 else "Female"
            st.write(f"**Prediction: {pred_class}**")
    except Exception as e:
        st.error(f"Error processing image or prediction: {e}")