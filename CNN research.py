import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

def create_model():
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(180, 180, 3)),  # Normalize the image
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # Output layer for red and green percentages
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (180, 180))  # Resize to 180x180
    img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
    return img_normalized


def process_image(image_path):
    img = cv2.imread(image_path)
    grid_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and green
    lower_red1 = np.array([0, 150, 50])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([170, 150, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([15, 100, 50])
    upper_green = np.array([105, 255, 255])

    # Create masks for red and green
    red_mask1 = cv2.inRange(grid_HSV, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(grid_HSV, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(grid_HSV, lower_green, upper_green)

    # Count non-zero pixels
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    total_color_pixels = red_pixels + green_pixels

    percentage_red = (red_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0
    percentage_green = (green_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0

    return percentage_red, percentage_green


def train_model(image_folder):
    image_paths = [Images]
    labels = ["dead","live","merge"]

    # Traverse through image_folder and subdirectories
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                red_percentage, green_percentage = process_image(image_path)

                image_paths.append(image_path)
                labels.append([red_percentage, green_percentage])

    # Debugging step
    print(f"Loaded {len(image_paths)} images.")

    if not image_paths:
        raise ValueError("No images found in the specified folder.")

    # Preprocess images
    images = np.array([preprocess_image(path) for path in image_paths])  # Shape: (num_samples, 180, 180, 3)
    labels = np.array(labels)  # Shape: (num_samples, 2)

    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    model = create_model()
    model.fit(images, labels, epochs=40, batch_size=32)

    # Save the trained model
    model.save("red_green_model_with_percentages.h5")
    print("Model saved as red_green_model_with_percentages.h5")

    return model


train_model(r"C:\MINI PROJECTS\Sem 5 for 22 November\Images")



