from flask import Flask, request, send_file
import numpy as np
import cv2
import dlib
import io
from scipy.spatial import distance
from PIL import Image
from skimage import io as skio
import os

app = Flask(__name__)
predictor_path = "shape_predictor_68_face_landmarks.dat"  # make sure this file is in your root folder

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

MAR_THRESHOLD = 0.3
ALPHA = 2.0  # Contrast
BETA = 50    # Brightness

def mouth_aspect_ratio(pts):
    D = distance.euclidean(pts[33], pts[51])
    D1 = distance.euclidean(pts[61], pts[67])
    D2 = distance.euclidean(pts[62], pts[66])
    D3 = distance.euclidean(pts[63], pts[65])
    mar = (D1 + D2 + D3) / (3 * D)
    return mar

def shape2np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def whiten_teeth(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        return None

    # Select largest face
    max_face = max(faces, key=lambda d: (d.bottom()-d.top())*(d.right()-d.left()))
    shape = predictor(image, max_face)
    pts = shape2np(shape)

    if mouth_aspect_ratio(pts) < MAR_THRESHOLD:
        return None  # mouth not open

    crop_img = image[max_face.top():max_face.bottom(), max_face.left():max_face.right()]
    mask = np.zeros((crop_img.shape[0], crop_img.shape[1]), np.uint8)

    mouth_pts = pts[60:] - np.array([max_face.left(), max_face.top()])
    cv2.fillConvexPoly(mask, mouth_pts, 255)

    blur_mask = cv2.GaussianBlur(mask, (21, 21), 11)
    overlay = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
    overlay[:, :, 3] = blur_mask

    # Brighten and contrast
    brightened = cv2.convertScaleAbs(overlay, alpha=ALPHA, beta=BETA)

    alpha = blur_mask.astype(float) / 255.0
    result = crop_img.copy()
    for c in range(3):
        result[:, :, c] = (1.0 - alpha) * crop_img[:, :, c] + alpha * brightened[:, :, c]

    output = image.copy()
    output[max_face.top():max_face.bottom(), max_face.left():max_face.right()] = result
    return output

@app.route('/')
def home():
    return 'Teeth Whitening API Running ✅'

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded.', 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = whiten_teeth(image)
    if result is None:
        return 'No face or open mouth detected.', 400

    _, buffer = cv2.imencode('.png', result)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
