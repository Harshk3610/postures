import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from posture_detection import process_frame  # assuming your code is in posture_detection.py

# Title
st.title('Sitting Posture Detection')

# Permission request and description
st.markdown("""
    **This app will request access to your webcam for posture detection.**

    This deployed model uses the live feed from the camera to detect the sitting posture.
""")

# Define the VideoTransformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.fps = 30  # Assume a standard fps if actual fps cannot be determined

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the frame
        output_frame = process_frame(img, self.fps)

        return output_frame

# Button to start the camera
if st.button('Use Camera'):
    webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

