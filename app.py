import streamlit as st
import cv2
import numpy as np
import face_recognition
import os

st.set_page_config(
    page_title="Face Recognition App",
    layout="centered"
)

st.title("Live Face Recognition")
st.write("Detect faces and display the person's name.")

# Main folder
KNOWN_FACES_DIR = "known_faces"

known_encodings = []
known_names = []

# =========================
# LOAD KNOWN FACES
# =========================
for person_name in os.listdir(KNOWN_FACES_DIR):

    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

    # Make sure it is a folder
    if os.path.isdir(person_folder):

        for filename in os.listdir(person_folder):

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(person_folder, filename)

                try:
                    image = face_recognition.load_image_file(image_path)

                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)

                except Exception as e:
                    st.warning(f"Could not load {image_path}")

# =========================
# CAMERA INPUT
# =========================
camera_image = st.camera_input("Open Camera")

if camera_image is not None:

    # Convert image
    bytes_data = camera_image.getvalue()
    np_array = np.frombuffer(bytes_data, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # =========================
    # DETECT FACES
    # =========================
    face_locations = face_recognition.face_locations(rgb_img)

    face_encodings = face_recognition.face_encodings(
        rgb_img,
        face_locations
    )

    # =========================
    # RECOGNIZE FACES
    # =========================
    for face_encoding, (top, right, bottom, left) in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown"

        if len(known_encodings) > 0:

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match_index = np.argmin(distances)

            # Smaller number = stricter recognition
            if distances[best_match_index] < 0.50:
                name = known_names[best_match_index]

        # Draw rectangle
        cv2.rectangle(
            img,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            3
        )

        # Draw name
        cv2.putText(
            img,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Convert to RGB for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(
        img_rgb,
        caption="Detected Faces",
        use_container_width=True
    )

    st.success(f"Faces detected: {len(face_locations)}")
