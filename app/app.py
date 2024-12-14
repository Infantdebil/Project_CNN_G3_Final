import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model("model.h5")

# Define class names (update with your own classes)
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

# App title
st.title("Image Classification App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize((128, 128))  # Resize to the input shape of your model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)
    probabilities = tf.nn.softmax(predictions[0])  # Convert logits to probabilities

    # Display predictions
    st.write("Prediction Results:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        st.write(f"{class_name}: {prob:.2%}")

