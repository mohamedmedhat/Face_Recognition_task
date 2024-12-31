import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import time
from helper import processing_img

model = load_model("./models/model.keras")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="wide")
st.title("Face Expression Recognition App")

st.markdown("""
    <style>
        body {
            background-color: #f0f0f5;
            font-family: 'Arial', sans-serif;
        }
        .header {
            color: #4CAF50;
            font-size: 3rem;
            font-weight: bold;
        }
        .description {
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: 1px solid #4CAF50;
            padding: 10px 20px;
            font-size: 1rem;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            padding-top: 20px;
            color: #777;
        }
        .image-frame {
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="description">
        Upload an image or record a video, and I will predict the emotion (happy, sad, angry, etc.)!
    </div>
""", unsafe_allow_html=True)

way = st.radio("Choose how to provide input:", ("Upload an image", "Record a video"))

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

        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.markdown("""
    <style>
        .uploaded-image {
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

        st.markdown(f'<img src="data:image/png;base64,{img}" class="uploaded-image" style="width:100%;" />', unsafe_allow_html=True)


        st.write(f"Predicted Emotion: {emotion}")
        st.success(f"Emotion: {emotion} detected!")

# elif way == "Record a video":
#     st.text("Click below to start video capture.")
#     start_button = st.button("Start Video Capture")

#     if start_button:
#         st.toast("Video capture started!")

#         video_file = st.camera_input("Take a picture")

#         if video_file:
#             frame = Image.open(video_file)
#             frame = np.array(frame)

#             processed_frame = processing_img(frame)
#             pred = model.predict(processed_frame)
#             emotion_idx = np.argmax(pred, axis=1)
#             emotion = emotion_labels[emotion_idx[0]]
            
#             st.image(frame, channels="RGB", caption=f"Predicted Emotion: {emotion}", use_column_width=True)
#             time.sleep(0.1)

# ! this didn't worked in the streamlit
elif way == "Record a video":
    st.text("Click below to start video capture.")
    start_button = st.button("Start Video Capture")

    if start_button:
        st.toast("Video capture started!")

        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = processing_img(frame)
            pred = model.predict(processed_frame)
            emotion_idx = np.argmax(pred, axis=1)
            emotion = emotion_labels[emotion_idx[0]]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", caption=f"Predicted Emotion: {emotion}", use_column_width=True)

            time.sleep(0.1)

        cap.release()

st.markdown("""
    <footer class="footer">
        Created by Mohamed Medhat Elnggar
    </footer>
    """, unsafe_allow_html=True)
