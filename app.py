import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Face Detection with Name", layout="centered")

st.title("Face Detection with Name")
st.write("Take a photo. The app will detect the face and show the selected name.")

person_name = st.text_input("Enter person's name", "Ghani")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    np_array = np.frombuffer(bytes_data, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            img,
            person_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Detected Face", use_container_width=True)
    st.success(f"Faces detected: {len(faces)}")
