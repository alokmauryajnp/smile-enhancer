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
    h, w, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return image

        landmarks = results.multi_face_landmarks[0]

        # Teeth landmarks: inner mouth (including top and bottom teeth area)
        teeth_indices = [
            78, 79, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14
        ]

        # Convert to image coordinates
        teeth_points = np.array([
            (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
            for i in teeth_indices
        ])

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [teeth_points], 255)

        # Convert to Lab color space to target luminance
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(img_lab)

        # Apply brightening only where mask is applied
        l = np.where(mask == 255, cv2.add(l, 25), l)
        updated_lab = cv2.merge((l, a, b))

        # Convert back to BGR
        result = cv2.cvtColor(updated_lab, cv2.COLOR_Lab2BGR)

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
