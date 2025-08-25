import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI issues

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("best (4).pt")

model = load_model()

st.title("🔍 PCB Defect Detection using YOLOv11")

# Upload image
uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array
    img_np = np.array(image)

    # Run inference
    results = model.predict(img_np, conf=0.25)

    # Get detections
    detections = results[0].boxes  # YOLOv11 detections
    num_defects = len(detections)  # count defects

    # Draw predictions
    res_plotted = results[0].plot()  # numpy array with boxes

    # Display results
    st.subheader("📊 Defect Detection Results")
    st.image(res_plotted, caption="Detected Defects with Confidence Scores", use_container_width=True)

    # Show number of detected defects
    st.success(f"✅ Number of defects detected: {num_defects}")

    # (Optional) List confidence scores
    if num_defects > 0:
        st.write("Confidence scores of detected defects:")
        for i, box in enumerate(detections):
            st.write(f"Defect {i+1}: {float(box.conf[0]):.2f}")
    else:
        st.info("No defects detected.")
