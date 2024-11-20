import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import cv2
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import ollama
from streamlit_d3graph import d3graph
from streamlit_drawable_canvas import st_canvas

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

def Chat():
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by Ollama")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        response = ollama.chat(model="llama3.2", messages=st.session_state.messages, stream=True)
        

        def stream_response():
            global incomingmsg
            incomingmsg = ""
            for chunk in response:
                part = chunk['message']['content']
                incomingmsg += part
                yield part
            
    
        st.chat_message("assistant").write(stream_response)
        st.session_state.messages.append({"role": "assistant", "content": incomingmsg})
            
def graph():
    # Initialize
    d3 = d3graph()
    # Load karate example
    adjmat, df = d3.import_example('karate')

    label = df['label'].values
    node_size = df['degree'].values

    d3.graph(adjmat)
    d3.set_node_properties(color=df['label'].values)
    d3.show()

    d3.set_node_properties(label=label, color=label, cmap='Set1')
    d3.show()

def painting():
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=400,
        width=600,
        drawing_mode=drawing_mode,
        key="canvas",
    )
if __name__ == "__painting__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    painting()


home_page = st.Page(home, title="Homepage")
info_page = st.Page(Chat, title="Chat")
graph_page = st.Page(graph, title="graph")
painting_page2 = st.Page(painting, title="Painting")


pg = st.navigation([home_page, info_page, graph_page, painting_page2])

pg.run()