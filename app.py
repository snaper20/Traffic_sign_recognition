import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('traffic_sign_model2.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    try:
        img = Image.open(image)
        img = img.resize((30, 30))  # Resize to the size expected by the model
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit app
st.title('Traffic Sign Recognition)

uploaded_file = st.file_uploader("Choose an image...", type=["ppm","jpg","png","jpeg"])

if uploaded_file is not None:
    # Preprocess the image
    img_array = preprocess_image(uploaded_file)
    
    if img_array is not None:
        # Predict the class of the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Predicted Class: {predicted_class}')
    else:
        st.error("Invalid image format. Please upload a valid image.")
