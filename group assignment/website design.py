import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    image = cv2.Canny(img, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)