from flask import Flask, request, send_file
from PIL import Image
import cv2
import numpy as np
import io
import tempfile

app = Flask(__name__)

def whiten_teeth(image_pil):
    # Convert PIL to OpenCV
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to HSV to isolate yellowish regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Mask yellowish areas (likely teeth stains)
    lower_yellow = np.array([10, 30, 80])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply whitening: Increase brightness in masked areas
    whitening_strength = 30
    image[mask > 0] = cv2.add(image[mask > 0], (whitening_strength, whitening_strength, whitening_strength))

    # Convert back to RGB and return as PIL image
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    
    try:
        # Open the uploaded image
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return f'Failed to read image: {e}', 400

    # Process the image
    processed_image = whiten_teeth(image)

    # Save image to memory for return
    buffer = io.BytesIO()
    processed_image.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='whitened.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
