import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import cv2
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


import numpy as np
if 'openpose' not in st.session_state:
    print("loading openpose")
    st.session_state['openpose'] =  OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


def video_frame_callback(frame):
    input_image = frame.to_ndarray(format="bgr24")

    output_image = cv2.Canny(input_image, 100, 200)
    output_image = output_image[:, :, None]
    output_image = np.concatenate([output_image, output_image, output_image], axis=2)
    image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png")


    # processor = st.session_state.openpose

    # output_image = processor.openpose(input_image)
    # output_image.save("test.png")
    return av.VideoFrame.from_ndarray(output_image, format="bgr24")

def home():
    st.subheader("Home")
 


    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

def chat():
    st.text("hi")
    st.button("Reset", type="primary")
    if st.button("Say hello"):
        st.write("Why hello there")
    else:
        st.write("Goodbye")


home_page = st.Page(home, title="Homepage")
info_page = st.Page(chat, title="chat")

pg = st.navigation([home_page, info_page])

pg.run()