import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import requests
from streamlit_lottie import st_lottie, st_lottie_spinner

# Load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Weather-themed Lottie (you can change the link if needed)
lottie_url_weather = "https://lottie.host/6de02fa5-b693-4e29-8b8b-8cb9e4bc2164/Qf26qLruod.json"  # Sample weather animation
lottie_weather = load_lottieurl(lottie_url_weather)

# Show welcome animation for 5 seconds only if loaded successfully
if lottie_weather:
    with st.container():
        st_lottie(lottie_weather, key="hello", height=300)
        st.markdown("<h3 style='text-align: center;'>Hello from the Weather Classification APP ☁️</h3>", unsafe_allow_html=True)
        time.sleep(5)
    st.empty()  # Clear the animation
else:
    st.warning("⚠️ Could not load the welcome animation. Please check your internet connection or the animation URL.")

# Main App
classes_names = ['DEW', 'Fogs Mog', 'Frost', 'Glaze', 'Hail', 'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sand Storm', 'Snow']

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('densenet_model.keras')
    return model

model = load_model()

# Sidebar
st.sidebar.title("About 🌧🤗")
st.sidebar.info("This app classifies weather conditions based on images.")

# Title and description
st.title("Weather Classifier 🌤🌧")
st.markdown("Upload an image and see if the AI can predict the weather condition correctly.")

# File uploader
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.warning("Please upload an image before clicking classify.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = classes_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.markdown(f"### Prediction: {predicted_class}")
    st.markdown(f"### Confidence: {confidence:.2f}%")

# Download animation
if st.button("Download"):
    with st_lottie_spinner(lottie_download, key="download"):
        time.sleep(5)
    st.balloons()
