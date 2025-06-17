from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to overlay transparent image
def overlay_transparent(background, overlay, x, y):
    bg = background.copy()
    h, w = overlay.shape[:2]

    if overlay.shape[2] != 4:
        raise ValueError("Overlay image must have 4 channels (RGBA).")

    if x + w > bg.shape[1] or y + h > bg.shape[0]:
        raise ValueError("Overlay image exceeds background bounds.")

    overlay_img = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3:] / 255.0
    bg_region = bg[y:y+h, x:x+w]
    blended = (1.0 - alpha_mask) * bg_region + alpha_mask * overlay_img
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    return bg

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    # Load user image
    face = cv2.imread(filepath)
    if face is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Load transparent white teeth overlay
    teeth = cv2.imread('white_teeth.png', cv2.IMREAD_UNCHANGED)
    if teeth is None or teeth.shape[2] != 4:
        return jsonify({'error': 'Teeth image not found or invalid format'}), 500

    # Resize and position (you can improve with face detection)
    teeth_resized = cv2.resize(teeth, (220, 80))  # adjust as needed
    x_pos, y_pos = 200, 330                      # adjust as needed

    try:
        result = overlay_transparent(face, teeth_resized, x_pos, y_pos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
    cv2.imwrite(result_path, result)

    return send_file(result_path, mimetype='image/png')

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Teeth Whitening App</title>
    <h1>Upload Image to Whiten Teeth</h1>
    <form method=post enctype=multipart/form-data action="/upload">
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
