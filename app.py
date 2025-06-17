from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
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

        # Teeth-related indices (inner mouth)
        teeth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312]
        teeth_points = np.array([
            (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
            for i in teeth_indices
        ])

        # Mask for polygon around teeth
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [teeth_points], 255)

        # Color filtering in HSV space for tooth tones
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 150])
        upper = np.array([60, 80, 255])
        color_mask = cv2.inRange(hsv, lower, upper)

        # Combine both masks
        teeth_mask = cv2.bitwise_and(mask, color_mask)

        # Apply whitening in Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l = np.where(teeth_mask == 255, np.clip(l + 15, 0, 255), l)
        updated_lab = cv2.merge((l.astype(np.uint8), a, b))
        result = cv2.cvtColor(updated_lab, cv2.COLOR_Lab2BGR)

        return result
# app route starts from here
@app.route('/')
def index():
    return 'Smile Whitening API is running!'

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = whiten_teeth(image)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, result)

    return send_file(temp_file.name, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
