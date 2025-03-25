TF_ENABLE_ONEDNN_OPTS=0
from flask import Flask, request, render_template
import cv2
import numpy as np
import io
from PIL import Image
import base64
import os
from tensorflow.keras.models import load_model  # type: ignore


app = Flask("Cell Classification")

# Load the trained model
model = load_model('red_green_model_with_percentages.h5')

def image_to_base64(image):
    """Convert an image to base64."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Save as PNG
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

def process_image(image_path):
    """Preprocess the image to prepare it for classification."""
    img = cv2.imread(image_path)
    
    # Resize the image to the model's expected input size (e.g., 180x180)
    img_resized = cv2.resize(img, (180, 180))  

    # Normalize the image (pixel values to [0, 1])
    img_normalized = img_resized / 255.0

    # Return the preprocessed image (with an added batch dimension)
    return np.expand_dims(img_normalized, axis=0)

def extract_color_percentages(image_path):
    """Extract red and green color percentages from the image."""
    img = cv2.imread(image_path)
    grid_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color ranges for red and green
    lower_red1 = np.array([0, 150, 50])
    upper_red1 = np.array([40, 255, 255])
    lower_red2 = np.array([160, 150, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([15, 100, 50])
    upper_green = np.array([105, 255, 255])

    # Create masks for red and green colors
    red_mask1 = cv2.inRange(grid_HSV, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(grid_HSV, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(grid_HSV, lower_green, upper_green)

    # Count non-zero pixels (colored pixels)
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    total_color_pixels = red_pixels + green_pixels

    # Calculate the percentage of red and green pixels
    percentage_red = (red_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0
    percentage_green = (green_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0

    # Normalize percentages
    total_percentage = percentage_red + percentage_green
    if total_percentage > 0:
        percentage_red = (percentage_red / total_percentage) * 100
        percentage_green = (percentage_green / total_percentage) * 100

    # Prepare processed image with red and green highlights
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_red = cv2.bitwise_and(grid_RGB, grid_RGB, mask=red_mask)
    res_green = cv2.bitwise_and(grid_RGB, grid_RGB, mask=green_mask)
    res_combined = cv2.add(res_red, res_green)

    # Convert processed image to base64
    processed_image = Image.fromarray(res_combined)
    processed_image_base64 = image_to_base64(processed_image)

    return percentage_red, percentage_green, processed_image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle the file upload and classification."""
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        if file:
            # Save the image temporarily
            img_path = 'uploaded_image.jpg'
            file.save(img_path)

            # Preprocess the image for the model (resize, normalize)
            img_preprocessed = process_image(img_path)

            # Make a prediction using the model
            prediction = model.predict(img_preprocessed)
            class_label = np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability
            class_confidence = np.max(prediction) * 100  # Get the prediction confidence

            # Extract the color percentages (red and green) and processed image
            percentage_red, percentage_green, processed_image_base64 = extract_color_percentages(img_path)

            # Optionally, you can also extract the base64 of the uploaded image for display in the result page
            original_image = Image.open(img_path)
            original_image_base64 = image_to_base64(original_image)

            # Remove the uploaded image after processing
            os.remove(img_path)

            # Render the result page with the prediction, color percentages, and images
            return render_template('result.html',
                                   original_image_data=original_image_base64,
                                   image_data=processed_image_base64,
                                   class_label=class_label,
                                   class_confidence=class_confidence,
                                   percentage_red=percentage_red,
                                   percentage_green=percentage_green)

    # Render the index page for GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
