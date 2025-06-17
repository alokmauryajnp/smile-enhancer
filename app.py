from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

def whiten_teeth(image):
    # Convert to RGB for Mediapipe
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, refine_connections=True,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return image  # No face detected

        # Get landmarks for the mouth area
        landmarks = results.multi_face_landmarks[0]
        mouth_indices = list(range(78, 88)) + list(range(308, 318))
        h, w, _ = image.shape
        mouth_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in mouth_indices]

        # Create a mask for the mouth area
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], 255)

        # Create whitening overlay
        whitened = image.copy()
        white_overlay = np.full_like(image, 255)

        # Blend only mouth area
        blended = cv2.addWeighted(whitened, 0.7, white_overlay, 0.3, 0)
        whitened[mask == 255] = blended[mask == 255]

        return whitened


@app.route('/')
def home():
    return jsonify({"status": "Smile Enhancer API is live!"})

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = whiten_teeth(image)

    # Save to temp file and return
    _, temp_path = tempfile.mkstemp(suffix=".jpg")
    cv2.imwrite(temp_path, result)
    return send_file(temp_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
