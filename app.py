import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace

st.set_page_config(page_title="Face Recognition", layout="centered")

st.title("Face Recognition App")
st.write("Take a photo and the app will show the person's name.")

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    st.error("Folder 'known_faces' not found.")
    st.stop()

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    np_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    temp_img_path = "captured_face.jpg"
    cv2.imwrite(temp_img_path, img)

    try:
        result = DeepFace.find(
            img_path=temp_img_path,
            db_path=KNOWN_FACES_DIR,
            model_name="Facenet",
            enforce_detection=False
        )

        name = "Unknown"

        if len(result) > 0 and len(result[0]) > 0:
            matched_path = result[0].iloc[0]["identity"]
            filename = os.path.basename(matched_path)
            name = os.path.splitext(filename)[0]

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.success(f"Person detected: {name}")

    except Exception as e:
        st.error("Face recognition failed.")
        st.write(e)
