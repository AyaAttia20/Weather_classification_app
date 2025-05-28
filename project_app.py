import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import requests

from streamlit_lottie import st_lottie, st_lottie_spinner

# --------------------- Load Lottie Animations --------------------- #
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_url_loading = "https://assets2.lottiefiles.com/private_files/lf30_j1adxtyb.json"  # Spinner

lottie_hello = load_lottieurl(lottie_url_hello)
lottie_loading = load_lottieurl(lottie_url_loading)

# --------------------- Sidebar Sections --------------------- #
st.sidebar.title("Weather Classifier 🌦️")

st.sidebar.header("📖 About")
st.sidebar.info("This app classifies weather conditions from uploaded images using a pre-trained CNN model.")

st.sidebar.header("🧠 How CNN Works")
st.sidebar.markdown("""
A Convolutional Neural Network (CNN) works by:
- Automatically learning features from images
- Using convolutional layers to extract patterns
- Classifying input based on learned patterns

It's especially powerful for image classification tasks like weather recognition.
""")

st.sidebar.header("📞 Contact Us")
st.sidebar.markdown("""
- 📧 Email: support@weatherai.com  
- 💻 GitHub: [WeatherAI Repo](https://github.com/AyaAttia20/Weather_classification_app)
""")

# --------------------- Lottie Intro Animation --------------------- #
st_lottie(lottie_hello, key="hello", height=200)

# --------------------- Model & Class Definitions --------------------- #
classes_names = ['DEW', 'Fogs Mog', 'Frost', 'Glaze', 'Hail', 'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sand Storm', 'Snow']

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('densenet_model.keras')
    return model

model = load_model()

# --------------------- Main App UI --------------------- #
st.title("Weather Classifier 🌤️🌧️")
st.markdown("Upload an image and let the AI predict the weather condition!")

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict on button click
    if st.button("Predict"):
        with st_lottie_spinner(lottie_loading, key="loading"):
            time.sleep(2)  # Simulate processing
            predictions = model.predict(img_array)
            predicted_class = classes_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.markdown(f"### ✅ Prediction: `{predicted_class}`")
        st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")
        st.snow()
else:
    st.warning("📁 Please upload an image before clicking Predict.")
