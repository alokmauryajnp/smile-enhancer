import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
import tempfile
import os

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh

def whiten_teeth(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return image  # No face detected

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        mouth_indices = list(range(78, 88)) + list(range(308, 318))
        mouth_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in mouth_indices]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], 255)

        white_overlay = np.full_like(image, 255)
        blended = cv2.addWeighted(image, 0.7, white_overlay, 0.3, 0)

        result = image.copy()
        result[mask == 255] = blended[mask == 255]

        return result

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    in_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(in_file.name)

    image = cv2.imread(in_file.name)
    output_image = whiten_teeth(image)

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(out_file.name, output_image)

    return send_file(out_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
