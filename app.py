from io import BytesIO
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from helper import processing_img

model = load_model("./models/model.keras")
emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}

st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="wide")
st.title("Face Expression Recognition App")

st.markdown("""
    <div style="font-size: 1.2rem; color: #555;">
        Upload an image or record a video, and I will predict the emotion (happy, sad, angry, etc.)!
    </div>
""", unsafe_allow_html=True)

way = st.radio("Choose how to provide input:", ("Upload an image", "Take an image"))

if way == "Upload an image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if image_file:
        progress = st.progress(0)
        st.write("Processing Image...")
        
        img = Image.open(image_file)
        img = np.array(img)
        processed_img = processing_img(img)
        
        progress.progress(50)
        
        pred = model.predict(processed_img)
        emotion_idx = np.argmax(pred, axis=1)
        emotion = emotion_labels[emotion_idx[0]]

        progress.progress(100)

        st.image(img, caption="Uploaded Image", use_container_width=False, width=270)
        st.success(f"Emotion: {emotion} detected!")

elif way == "Take an image":
    st.text("Click below to start take an image.")
    start_button = st.button("Start Image Capture")

    if start_button:
        st.toast("Image capture started!")

        video_file = st.camera_input("Take a picture")

        if video_file is not None:
            frame = Image.open(BytesIO(video_file)) 
            frame = np.array(frame)

            processed_frame = processing_img(frame)
            pred = model.predict(processed_frame)
            emotion_idx = np.argmax(pred, axis=1)
            emotion = emotion_labels[emotion_idx[0]]

            st.image(frame, caption=f"Predicted Emotion: {emotion}", width=270)
            st.success(f"Emotion: {emotion} detected!")

st.markdown("""
    <footer style="font-size: 14px; text-align: center; padding-top: 20px; color: #777;">
        Created by Mohamed Medhat Elnggar
    </footer>
""", unsafe_allow_html=True)
