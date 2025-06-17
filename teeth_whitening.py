import cv2
import numpy as np

# Step 1: Load the input image
input_path = 'ggg.png'       # Replace with your input image filename
output_path = 'result.png'   # Output file name

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at path: {input_path}")

# Step 2: Convert image to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)

# Step 3: Create a mask for yellow teeth
# Yellow has higher B channel value; threshold to isolate yellowish teeth
_, teeth_mask = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY)

# Step 4: Refine the mask using morphological operations
kernel = np.ones((3, 3), np.uint8)
teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 5: Whitening effect - brighten (L+) and reduce yellowness (B-)
l_whitened = cv2.add(l, 20)                # Increase lightness
b_less_yellow = cv2.subtract(b, 30)        # Reduce yellowness

# Step 6: Merge modified channels back
lab_whitened = cv2.merge((l_whitened, a, b_less_yellow))
bgr_whitened = cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)

# Step 7: Apply mask only to teeth area
teeth_mask_3ch = cv2.merge([teeth_mask]*3)  # Convert to 3-channel mask
teeth_mask_bool = teeth_mask_3ch.astype(bool)

# Copy only the whitened teeth into original image
output = image.copy()
output[teeth_mask_bool] = bgr_whitened[teeth_mask_bool]

# Step 8: Save the result
cv2.imwrite(output_path, output)
print(f"✅ Teeth whitening completed. Saved as '{output_path}'.")
