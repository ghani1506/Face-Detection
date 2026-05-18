import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Live Face Detection", layout="centered")

st.title("Live CNN Face Detection")
st.write("Use your phone camera for real-time face detection.")

# OpenCV pretrained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open phone camera
camera_image = st.camera_input("Open Camera")

if camera_image is not None:

    # Convert image
    bytes_data = camera_image.getvalue()
    np_array = np.frombuffer(bytes_data, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Detected Faces", use_container_width=True)

    st.success(f"Faces detected: {len(faces)}")
