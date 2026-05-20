import streamlit as st
import cv2
import numpy as np
import face_recognition
import os

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Face Recognition App",
    layout="centered"
)

st.title("Live Face Recognition")
st.write("Detect faces and display the person's name.")

# =========================
# KNOWN FACES FOLDER
# =========================
KNOWN_FACES_DIR = "known_faces"

# Check folder exists
if not os.path.exists(KNOWN_FACES_DIR):
    st.error("Folder 'known_faces' not found.")
    st.stop()

known_encodings = []
known_names = []

# =========================
# LOAD KNOWN FACES
# =========================
for filename in os.listdir(KNOWN_FACES_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:

                known_encodings.append(encodings[0])

                # File name becomes person's name
                name = os.path.splitext(filename)[0]

                known_names.append(name)

        except Exception as e:
            st.warning(f"Could not load {filename}")

# =========================
# CAMERA
# =========================
camera_image = st.camera_input("Take a picture")

if camera_image is not None:

    # Convert uploaded image
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

    # Convert image for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(
        img_rgb,
        caption="Detected Faces",
        use_container_width=True
    )

    st.success(f"Faces detected: {len(face_locations)}")
