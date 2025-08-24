import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained YOLOv11 model
model = YOLO("best (4).pt")  # ensure best.pt is in the same folder

st.title("🔍 PCB Defect Detection App")
st.write("Upload an image of a PCB to detect defects.")

# File uploader
uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded PCB Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)

    # Run YOLO inference
    results = model(img_array)

    # Draw detections on the image
    annotated_img = results[0].plot()  # ultralytics provides plot()

    st.image(annotated_img, caption="Defect Detection Results", use_column_width=True)

    # Show predictions in text
    st.subheader("Detections:")
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        st.write(f"- {model.names[cls_id]} ({conf:.2f})")
