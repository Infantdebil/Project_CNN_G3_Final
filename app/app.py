import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import csv

# Load and compile the saved model
model = tf.keras.models.load_model("model_p.h5")
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

Our self-trained object detection model is designed to classify images into the following ten categories: **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, and **truck**.

The model was trained using the **CIFAR-10 dataset**, and all images were resized to **64x64** with bicubic interpolation. Augmentations were applied during training to improve generalization.

The model achieved an accuracy of **87% on training validation**, demonstrating its reliability in classifying objects within this dataset. This project was collaboratively developed by **Auréle, Enrique, and Paul**.
""")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
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
        st.write("Please provide feedback if the classification is incorrect or if the image does not belong to the classification categories.")

        feedback_options = class_names + ["Not part of classification"]
        correct_label = st.selectbox("Select the correct label or indicate that it's not part of the classification:", feedback_options)

        if st.button("Submit Feedback"):
            # Save the image and feedback to a folder
            feedback_image_path = os.path.join(feedback_folder, uploaded_file.name)
            image.save(feedback_image_path)
            
            # Save feedback to CSV
            with open(feedback_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([uploaded_file.name, correct_label])
            
            if correct_label == "Not part of classification":
                st.success("Thank you for your feedback! Your input has been saved and will help identify out-of-scope images.")
            else:
                st.success("Thank you for your feedback! Your input has been saved and will be considered during the next retraining of the model.")
