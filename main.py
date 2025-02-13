import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load model with error handling
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
current_working_directory = os.path.dirname(os.path.abspath(__file__))
model_path = f"{current_working_directory}/trained_model/class_model.keras"
model = load_model(model_path)

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        display_image = image.resize((100, 100))
        st.image(display_image)
    with col2:
        if st.button('Classify'):
            if model is not None:
                try:
                    with st.spinner('Classifying...'):
                        # Preprocess the uploaded image
                        img_array = preprocess_image(image)

                        # Make a prediction using the pre-trained model
                        result = model.predict(img_array)
                        
                        predicted_class_index = np.argmax(result)
                        prediction = class_names[predicted_class_index]
                        confidence = float(result[0][predicted_class_index]) * 100  # Convert to percentage

                        st.success(f'Prediction: {prediction}')
                        st.info(f'Confidence: {confidence:.2f}%')  # Format to 2 decimal places
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.error("Model is not loaded.")