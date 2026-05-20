import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# =====================================================
# PAGE SETTINGS
# =====================================================
st.set_page_config(
    page_title="Face Recognition App",
    layout="centered"
)

st.title("Face Recognition App")
st.write("Upload known face photos, then take a camera photo to detect and show the person's name.")

# =====================================================
# SETTINGS
# =====================================================
KNOWN_FACES_DIR = "known_faces"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# =====================================================
# LOAD FACE DETECTOR
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_name_from_file(filename):
    """
    Example:
    known_faces/Ghani.jpg -> Ghani
    known_faces/Ali.png -> Ali
    """
    return os.path.splitext(os.path.basename(filename))[0]


def load_known_faces():
    """
    Loads all images inside known_faces folder.
    The filename becomes the person's name.
    Example:
    known_faces/Ghani.jpg will display Ghani
    """
    known_face_images = []
    known_face_names = []

    files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for file in files:
        path = os.path.join(KNOWN_FACES_DIR, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            continue

        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        known_face_images.append(face_roi)
        known_face_names.append(get_name_from_file(file))

    return known_face_images, known_face_names


def recognize_face(test_face, known_face_images, known_face_names):
    """
    Simple OpenCV template matching.
    Lower score = better match.
    This is lightweight and deploys more easily than TensorFlow/DeepFace.
    """
    if len(known_face_images) == 0:
        return "Unknown", None

    test_face = cv2.resize(test_face, (200, 200))

    best_name = "Unknown"
    best_score = float("inf")

    for known_face, name in zip(known_face_images, known_face_names):
        score = np.mean((test_face.astype("float") - known_face.astype("float")) ** 2)

        if score < best_score:
            best_score = score
            best_name = name

    # You may adjust this.
    # Lower = stricter, higher = more forgiving.
    threshold = 4500

    if best_score < threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Known Faces")

uploaded_files = st.sidebar.file_uploader(
    "Upload known face images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(KNOWN_FACES_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success("Known face image(s) uploaded. Refresh or take a photo.")

known_files = [
    f for f in os.listdir(KNOWN_FACES_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

if known_files:
    st.sidebar.write("Current known faces:")
    for f in known_files:
        st.sidebar.write(f"- {get_name_from_file(f)}")
else:
    st.sidebar.warning("No known faces yet. Upload images first.")

# =====================================================
# LOAD KNOWN FACE DATA
# =====================================================
known_face_images, known_face_names = load_known_faces()

if len(known_face_names) == 0:
    st.warning("No usable known face found. Please upload a clear front-facing face photo.")
else:
    st.info(f"Loaded {len(known_face_names)} known face(s): {', '.join(known_face_names)}")

# =====================================================
# CAMERA INPUT
# =====================================================
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
        face_roi = gray[y:y+h, x:x+w]

        name, score = recognize_face(
            face_roi,
            known_face_images,
            known_face_names
        )

        label = name if score is None else f"{name}"

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(
        img_rgb,
        caption="Detected Face with Name",
        use_container_width=True
    )

    st.success(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        st.warning("No face detected. Try better lighting and face the camera.")
