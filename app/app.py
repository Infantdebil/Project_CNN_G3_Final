import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("model_p.h5")

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# App title and introduction
st.title("Image Classification CNN App")
st.image("header_image.png", use_column_width=True)  # Replace with your header image

st.write("""
# Custom Object Detection Model

Our self-trained object detection model is designed to classify images into the following ten categories: **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, and **truck**.

The model was trained using the **CIFAR-10 dataset**, which comprises 60,000 32x32 color images, evenly distributed across 10 classes (6,000 images per class). The dataset is structured with 50,000 training images and 10,000 test images. Training data is divided into five batches of 10,000 images each, with some variation in class distribution per batch. The test batch is balanced, containing exactly 1,000 images per class.

Our model achieved an accuracy of **87% on training validation**, demonstrating its reliability in classifying objects within this dataset. This project was collaboratively developed by **Auréle, Enrique, and Paul**.
""")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize((64, 64))  # Resize to the input shape of CIFAR-10
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)
    probabilities = tf.nn.softmax(predictions[0]).numpy()  # Convert logits to probabilities

    # Display predictions
    st.write("Prediction Results:")
    for class_name, prob in zip(class_names, probabilities):
        st.write(f"{class_name}: {prob:.2%}")

    # Determine the highest prediction and confidence
    max_prob = np.max(probabilities)
    max_index = np.argmax(probabilities)
    second_max_prob = sorted(probabilities, reverse=True)[1]
    
    # Logic to decide if the model is sure or unsure
    if max_prob > second_max_prob * 1.25:
        st.write(f"**The model predicts the image is: {class_names[max_index]} with {max_prob:.2%} confidence.**")
    else:
        st.write("**The model is not confident enough to make a prediction.**")
