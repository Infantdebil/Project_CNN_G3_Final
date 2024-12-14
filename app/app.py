import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load and compile the saved model
model = tf.keras.models.load_model("model_p.h5")
model.compile(optimizer='AdamW', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# App title and introduction
st.title("Image Classification CNN App")
st.image("header_image.png", use_container_width=True)  # Updated parameter for deprecation warning

st.write("""
# Custom Object Detection Model

Our self-trained object detection model is designed to classify images into the following ten categories: **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, and **truck**.

The model was trained using the **CIFAR-10 dataset**, and all images were resized to **64x64** with bicubic interpolation. Augmentations were applied during training to improve generalization.

The model achieved an accuracy of **87% on training validation**, demonstrating its reliability in classifying objects within this dataset. This project was collaboratively developed by **Auréle, Enrique, and Paul**.
""")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image and results side by side
    col1, col2 = st.columns([1, 2])  # Adjust column widths if needed

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    with col2:
        # Preprocess the image (add padding and resize to 64x64)
        def preprocess_image(image):
            # Ensure the image is RGB
            image = image.convert('RGB')
            # Add padding to make the image square
            image = ImageOps.pad(image, (64, 64), method=Image.BICUBIC, color=(0, 0, 0))
            # Convert to numpy array
            img_array = np.array(image).astype('float32') / 255.0  # Normalize pixel values
            # Add batch dimension
            return np.expand_dims(img_array, axis=0)
        
        img_array = preprocess_image(image)

        # Get predictions
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()  # Convert logits to probabilities

        # Extract top 3 predictions
        sorted_indices = np.argsort(probabilities)[::-1]  # Indices of predictions sorted in descending order
        top_3_indices = sorted_indices[:3]
        top_3_classes = [class_names[i] for i in top_3_indices]
        top_3_probs = [probabilities[i] for i in top_3_indices]

        # Display prediction results
        max_prob = top_3_probs[0]
        max_index = top_3_indices[0]
        second_max_prob = top_3_probs[1]

        # Confident or unsure logic
        if max_prob > second_max_prob * 1.25:
            st.write(f"**The model predicts the image is: {class_names[max_index]} with {max_prob:.2%} confidence.**")
        else:
            st.write("**The model is not confident enough to make a prediction.**")
            st.write("Cant decide betweenthe following Prediction Results:")
            for class_name, prob in zip(top_3_classes, top_3_probs):
                st.write(f"{class_name}: {prob:.2%}")
