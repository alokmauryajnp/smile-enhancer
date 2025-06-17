from flask import Flask, request, send_file
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import io

app = Flask(__name__)

def detect_and_whiten_teeth(image_pil):
    # Convert PIL to OpenCV
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return image_pil  # Return original if no face detected

    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Mouth landmark indices (upper and lower lips)
    mouth_indices = list(range(78, 88)) + list(range(308, 318))

    for face_landmarks in results.multi_face_landmarks:
        points = [(int(landmark.x * width), int(landmark.y * height)) for i, landmark in enumerate(face_landmarks.landmark) if i in mouth_indices]
        if points:
            points = np.array(points)
            cv2.fillPoly(mask, [points], 255)

    # Apply whitening only to masked area
    mouth_area = cv2.bitwise_and(image, image, mask=mask)
    enhanced = image.copy()
    enhanced[mask > 0] = cv2.add(enhanced[mask > 0], (40, 40, 40))

    result = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return f'Failed to read image: {e}', 400

    processed_image = detect_and_whiten_teeth(image)

    buffer = io.BytesIO()
    processed_image.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='whitened.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
