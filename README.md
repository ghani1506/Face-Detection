# Face Recognition Streamlit App

This is a fresh-start face detection and simple name recognition app.

## Folder Structure

```text
face_recognition_fresh_start/
├── app.py
├── requirements.txt
├── known_faces/
│   ├── Ghani.jpg
│   ├── Ali.jpg
│   └── Huzaira.jpg
└── .streamlit/
    └── config.toml
```

## Important

Put known face photos inside the `known_faces` folder.

Example:

```text
known_faces/Ghani.jpg
```

The filename becomes the displayed name:

```text
Ghani.jpg -> Ghani
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This version avoids DeepFace, TensorFlow, dlib, and face_recognition because those packages often fail on Streamlit Cloud.

It uses OpenCV only, so it is much easier to deploy.

For best results:
- Use clear front-facing photos.
- Use one face per known image.
- Use good lighting.
- Name each file properly, e.g. `Ghani.jpg`.
