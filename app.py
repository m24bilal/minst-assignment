import time
import subprocess
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def start_fastapi():

    process = subprocess.Popen([
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ])

    time.sleep(5)

    return process


fastapi_process = start_fastapi()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢")
st.title("🔢 MNIST Digit Predictor")


@st.cache_resource
def get_model():
    return load_model("mnist.keras")

model = get_model()

uploaded_file = st.file_uploader("Digit image upload karo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = image.convert("L")            # grayscale
            img = img.resize((28, 28))

            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28)  # <-- (1, 28, 28)

            prediction = model.predict(img_array)
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"Predicted Digit: {predicted_digit}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

