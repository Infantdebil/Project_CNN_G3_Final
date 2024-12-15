import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import csv
import os
import tensorflow as tf
import streamlit as st

# Debug current directory
st.write("Current Working Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir("."))

# Load model dynamically
model_path = os.path.join(os.path.dirname(__file__), "model_p.h5")
st.write(f"Loading model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create a folder for saving feedback data
feedback_folder = "feedback_data"
os.makedirs(feedback_folder, exist_ok=True)
feedback_file = os.path.join(feedback_folder, "feedback.csv")

# App title and introduction
st.title("Image Classification CNN App")
st.image("header_image.png", use_container_width=True)

st.write("""
# Custom Object Detection Model

This self-trained model classifies images into the following categories: 
**airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck**.

The model uses the CIFAR-10 dataset and resizes all images to **64x64** pixels for compatibility.

Feedback is used to retrain and improve the model in future updates.
""")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Layout: side-by-side display for the image and predictions
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Preprocess the image
        def preprocess_image(image):
            image = image.convert('RGB')
            image = ImageOps.pad(image, (64, 64), method=Image.BICUBIC, color=(0, 0, 0))
            img_array = np.array(image).astype('float32') / 255.0
            return np.expand_dims(img_array, axis=0)
        
        img_array = preprocess_image(image)

        # Get predictions
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        sorted_indices = np.argsort(probabilities)[::-1]

        # Display predictions
        max_index = sorted_indices[0]
        max_prob = probabilities[max_index]
        second_max_prob = probabilities[sorted_indices[1]]

        st.write("**Prediction Results:**")
        if max_prob > second_max_prob * 1.25:
            st.write(f"**The model predicts the image is: {class_names[max_index]} with {max_prob:.2%} confidence.**")
        else:
            st.write("**The model is not confident enough to make a prediction.**")
            st.write("Top 3 Prediction Results:")
            for i in sorted_indices[:3]:
                st.write(f"{class_names[i]}: {probabilities[i]:.2%}")

        # Feedback section
        st.write("### Feedback")
        st.write("If the classification is incorrect or the image does not belong to any class, please let us know!")

        # Add "Not part of classification" option
        feedback_options = class_names + ["Not part of classification"]
        correct_label = st.selectbox("Select the correct label or indicate it's not part of the classification:", feedback_options)

        if st.button("Submit Feedback"):
            # Save the image and feedback to a folder
            feedback_image_path = os.path.join(feedback_folder, uploaded_file.name)
            image.save(feedback_image_path)
            
            # Save feedback to CSV
            with open(feedback_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([uploaded_file.name, correct_label])
            
            if correct_label == "Not part of classification":
                st.success("Thank you for your feedback! This will help us identify out-of-scope images.")
            else:
                st.success("Thank you for your feedback! Your input will be considered in the next model retraining.")
