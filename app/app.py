import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import csv

# ===========================
# Define Paths and Load Model
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_p.h5")
HEADER_IMAGE_PATH = os.path.join(BASE_DIR, "header_image.png")
FEEDBACK_FOLDER = os.path.join(BASE_DIR, "feedback_data")
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
FEEDBACK_FILE = os.path.join(FEEDBACK_FOLDER, "feedback.csv")

# Load model dynamically
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ===========================
# App Introduction
# ===========================
st.title("Image Classification CNN App")

# Display header image
try:
    st.image(HEADER_IMAGE_PATH, use_container_width=True)
except Exception as e:
    st.error(f"Error loading header image: {e}")

st.write("""
# Custom Object Detection Model
This model classifies images into the following categories:  
**airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck**.  

The model uses the CIFAR-10 dataset and resizes all images to **64x64** pixels for 
compatibility. The model achieved an accuracy of **87% on training validation**, 
demonstrating its reliability in classifying objects within this dataset. 
This project was collaboratively developed by **Auréle, Enrique, and Paul**.
          
Feedback is used to retrain and improve the model in future updates.
""")

# ===========================
# Image Upload and Processing
# ===========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    image = image.convert('RGB')
    image = ImageOps.pad(image, (64, 64), method=Image.BICUBIC, color=(0, 0, 0))
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    # Layout: image on left, predictions on right
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        try:
            # Preprocess and predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            probabilities = tf.nn.softmax(predictions[0]).numpy()
            sorted_indices = np.argsort(probabilities)[::-1]

            # Display prediction
            max_index = sorted_indices[0]
            max_prob = probabilities[max_index]
            second_max_prob = probabilities[sorted_indices[1]]

            st.write("### Prediction Results")
            if max_prob > second_max_prob * 1.25:
                st.markdown(f"""
                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">
                             The model is quite sure that your image represents a: {class_names[max_index]}
                    </div>
                    <div style="font-size: 18px; font-weight: normal;">
                        with <b>{max_prob:.2%}</b> confidence.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.write("**The model is not confident enough to make a prediction.**")
                st.write("Top 3 Prediction Results:")
                for i in sorted_indices[:3]:
                    st.write(f"{class_names[i]}: {probabilities[i]:.2%}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # ===========================
    # Feedback Section
    # ===========================
    st.write("### Feedback")
    st.write("If the classification is incorrect or the image does not belong to any class, please let us know!")
    
    feedback_options = class_names + ["Not part of classification"]
    correct_label = st.selectbox("Select the correct label or indicate it's not part of the classification:", feedback_options)
    
    if st.button("Submit Feedback"):
        try:
            # Save feedback image
            feedback_image_path = os.path.join(FEEDBACK_FOLDER, uploaded_file.name)
            image.save(feedback_image_path)

            # Log feedback
            with open(FEEDBACK_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([uploaded_file.name, correct_label])
            
            if correct_label == "Not part of classification":
                st.success("Thank you for your feedback! This will help identify out-of-scope images.")
            else:
                st.success("Thank you for your feedback! Your input will be used to retrain the model in future updates.")
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")

# ===========================
# Debugging Information (Optional)
# ===========================
# st.write("**Debugging Information:**")
# st.write(f"Current Working Directory: {os.getcwd()}")
# st.write("Files in Directory:", os.listdir(BASE_DIR))
