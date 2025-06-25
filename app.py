import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = load_model("gender_model.h5")

# Load and prep image
def load_and_prep(image, img_shape=224):
    image = image.resize((img_shape, img_shape))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]  # remove alpha if present
    image = image / 255.
    return image

# Streamlit UI
st.title("Gender Classifier")
st.write("Upload a face image to predict gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    image = load_and_prep(img)
    pred = model.predict(np.expand_dims(image, axis=0))
    pred_class = "Female" if tf.round(pred)[0][0] == 1 else "Male"
    st.write(f"**Prediction: {pred_class}**")
