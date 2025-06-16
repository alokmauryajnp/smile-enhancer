from flask import Flask, request, send_file
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)

# @app.route('/process-image', methods=['POST'])
@app.route("/", methods=["GET"])
def home():
    return "Smile enhancer is live!"

def process_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Whitening logic: increase brightness of full image (demo)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    result_img = Image.fromarray(result)
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')
