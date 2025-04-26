import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load your trained YOLOv11 fire detection model
model = YOLO("best.pt")  # Ensure best.pt is present in your working directory

# Streamlit page config
st.set_page_config(page_title="Fire Detection using YOLOv11", layout="wide")
st.title("🔥 Fire Detection using YOLOv11")

# Sidebar: Confidence Slider
st.sidebar.header("Select Model Confidence Value")
confidence = st.sidebar.slider("Confidence Threshold", min_value=10, max_value=100, value=40) / 100

# Sidebar: Source Selection
st.sidebar.header("Image/Video Config")
source_type = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam"])

# Sidebar: File Upload
uploaded_file = None
if source_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
elif source_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video...", type=["mp4", "mov", "avi"])

# Detection function
def detect_fire(image, conf_thresh):
    results = model.predict(image, conf=conf_thresh)
    boxes = results[0].boxes
    result_img = results[0].plot()
    return result_img, boxes, results[0]

# === Image Detection ===
if uploaded_file and source_type == "Image":
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.sidebar.button("🔍 Detect Objects"):
        with st.spinner("Detecting fire..."):
            result_image, boxes, result_obj = detect_fire(image, confidence)

            if boxes is not None and len(boxes) > 0:
                st.subheader("Detected Image with Bounding Boxes")
                st.image(result_image, use_container_width=True)

                # Display detected class labels
                class_ids = boxes.cls.tolist()
                detected_classes = [model.names[int(cls_id)] for cls_id in class_ids]
                st.success(f"✅ Detected Classes: {', '.join(detected_classes)}")
            else:
                st.warning("⚠️ No fire detected in the image. Try lowering the confidence threshold.")

# === Video Detection ===
if uploaded_file and source_type == "Video":
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.subheader("Uploaded Video")
    st.video(video_path)

    if st.sidebar.button("🎥 Detect Fire in Video"):
        st.info("Processing video... this might take a while.")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_path = "output_video.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=confidence)
            frame = results[0].plot()
            out.write(frame)

        cap.release()
        out.release()
        st.success("Video processing complete.")
        st.video(out_path)

# === Webcam Detection ===
if source_type == "Webcam":
    st.subheader("Webcam Live Fire Detection")

    start_cam = st.sidebar.button("Start Webcam")
    stop_cam = st.sidebar.button("Stop Webcam")

    if start_cam:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not read from webcam.")
                break

            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

            if stop_cam:
                break

        cap.release()
        cv2.destroyAllWindows()