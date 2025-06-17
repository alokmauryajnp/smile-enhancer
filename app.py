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

        # More comprehensive teeth landmark indices
        upper_teeth_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lower_teeth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
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
        
        # Dilate mask slightly to cover entire teeth area
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth mask edges for better blending
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        mask = mask.astype(np.float32) / 255.0  # Convert to 0-1 range
        
        # Convert to HSV color space (better for color manipulation)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply whitening effect only to masked areas
        v_whitened = v.copy()
        
        # Increase value (brightness) and reduce saturation for whitening effect
        v_whitened = np.where(mask > 0, 
                             np.clip(v * (1.0 + 0.3 * mask), 0, 255), 
                             v)
        
        s_whitened = np.where(mask > 0, 
                             np.clip(s * (1.0 - 0.5 * mask), 0, 255), 
                             s)
        
        # Merge channels back
        whitened_hsv = cv2.merge((h, s_whitened.astype(np.uint8), v_whitened.astype(np.uint8)))
        result = cv2.cvtColor(whitened_hsv, cv2.COLOR_HSV2BGR)
        
        # Blend with original image for natural look
        result = cv2.addWeighted(image, 1 - mask, result, mask, 0)
        
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
