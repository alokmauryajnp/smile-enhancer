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

        # Facial landmarks
        landmarks = results.multi_face_landmarks[0]

        # Approximate upper and lower teeth landmark indices
        upper_teeth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lower_teeth_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        # Convert to pixel coordinates
        upper_teeth_points = np.array([
            (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
            for i in upper_teeth_indices
        ])
        lower_teeth_points = np.array([
            (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
            for i in lower_teeth_indices
        ])

        # Create teeth mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [upper_teeth_points], 255)
        cv2.fillPoly(mask, [lower_teeth_points], 255)

        # Smooth mask to blend naturally
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # Brighten the L channel
        l_whitened = l.copy()
        l_whitened = np.where(mask > 0, np.clip(l + 25, 0, 255), l)

        # Merge and convert back
        updated_lab = cv2.merge((l_whitened.astype(np.uint8), a, b))
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
