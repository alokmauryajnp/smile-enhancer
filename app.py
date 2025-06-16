from flask import Flask, request, send_file
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Smile enhancer is live!"

@app.route("/process-image", methods=["POST"])
def process_image():
    print("Processing image request...")

    if 'image' not in request.files:
        print("⚠️ No image found in request.files")
        return "No image uploaded", 400

    file = request.files['image']
    print("✅ Image received:", file.filename)

    img = Image.open(file.stream).convert('RGB')

    # Basic whitening filter
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    result_img = Image.fromarray(result)
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')
