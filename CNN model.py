# import tensorflow as tf
# from tensorflow.keras import layers, models  # type: ignore
# import numpy as np
# import cv2
# import os

# # Model Architecture
# def create_model():
#     model = models.Sequential([
#         layers.Rescaling(1.0 / 255, input_shape=(180, 180, 3)),  # Normalize the image
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(2)  # Output layer: Two neurons (red and green percentages)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#     return model


# # Preprocess Image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img_resized = cv2.resize(img, (180, 180))  # Resize to 180x180
#     img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
#     return img_normalized


# # Process Image for Red/Green Percentages
# def process_image(image_path):
#     img = cv2.imread(image_path)
#     grid_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Define HSV ranges for red and green
#     lower_red1 = np.array([0, 150, 50])
#     upper_red1 = np.array([30, 255, 255])
#     lower_red2 = np.array([170, 150, 50])
#     upper_red2 = np.array([180, 255, 255])

#     lower_green = np.array([15, 100, 50])
#     upper_green = np.array([105, 255, 255])

#     # Create masks for red and green
#     red_mask1 = cv2.inRange(grid_HSV, lower_red1, upper_red1)
#     red_mask2 = cv2.inRange(grid_HSV, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(red_mask1, red_mask2)

#     green_mask = cv2.inRange(grid_HSV, lower_green, upper_green)

#     # Count non-zero pixels
#     red_pixels = cv2.countNonZero(red_mask)
#     green_pixels = cv2.countNonZero(green_mask)
#     total_color_pixels = red_pixels + green_pixels

#     percentage_red = (red_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0
#     percentage_green = (green_pixels / total_color_pixels) * 100 if total_color_pixels > 0 else 0

#     return percentage_red, percentage_green


# # Train Model
# def train_model(image_folder):
#     image_paths = []  # Initialize as empty list
#     labels = []  # Initialize as empty list

#     # Load all images and calculate their corresponding red/green percentages
#     for filename in os.listdir(image_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(image_folder, filename)
#             red_percentage, green_percentage = process_image(image_path)

#             image_paths.append(image_path)  # Add image path
#             labels.append([red_percentage, green_percentage])  # Add percentages as labels

#     # Debugging step
#     print(f"Loaded {len(image_paths)} images.")

#     if not image_paths:
#         raise ValueError("No images found in the specified folder.")

#     # Preprocess images
#     images = np.array([preprocess_image(path) for path in image_paths])  # Shape: (num_samples, 180, 180, 3)
#     labels = np.array(labels)  # Shape: (num_samples, 2)

#     # Add batch dimension to images (TensorFlow compatibility)
#     images = np.expand_dims(images, axis=0)

#     print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

#     # Verify shapes
#     if images.shape[0] != labels.shape[0]:
#         raise ValueError("Mismatch between the number of images and labels.")

#     # Create and train the model
#     model = create_model()
#     model.fit(images, labels, epochs=10, batch_size=32)

#     # Save the trained model
#     model.save("red_green_model.h5")
#     print("Model saved as red_green_model.h5")

#     return model


# if __name__ == "__main__":
#     train_model(r"I:\20 nov- Without model code\Images")

# import tensorflow as tf
# from tensorflow.keras import layers, models  # type: ignore
# import numpy as np


# # Model Architecture
# def create_model():
#     model = models.Sequential([
#         layers.Rescaling(1.0 / 255, input_shape=(180, 180, 3)),  # Normalize the image
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(3, activation='softmax')  # Output layer for 3 classes (live, dead, merge)
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# # Train Model
# def train_model(image_folder):
#     # Load dataset from directory
#     dataset = tf.keras.utils.image_dataset_from_directory(
#         image_folder,
#         labels="inferred",           # Automatically infer labels from subdirectory names
#         label_mode="categorical",    # For multi-class classification
#         image_size=(180, 180),       # Resize all images to 180x180
#         batch_size=32
#     )
#     class_names = dataset.class_names
#     print(f"Classes found: {class_names}")

#     # Split the dataset into training and validation
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset = dataset.take(train_size)
#     val_dataset = dataset.skip(train_size)

#     # Prefetch data for performance optimization
#     train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#     # Create and train the model
#     model = create_model()
#     model.fit(train_dataset, validation_data=val_dataset, epochs=10)

#     # Save the trained model
#     model.save("red_green_model.h5")
#     print("Model saved as red_green_model.h5")

#     return model


# # Run the training
# train_model(r"I:\20 nov- Without model code\Images")



import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os


# Model Architecture
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


# Preprocess Image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (180, 180))  # Resize to 180x180
    img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
    return img_normalized


# Process Image for Red/Green Percentages
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


# Train Model
def train_model(image_folder):
    image_paths = []
    labels = []

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

    # Create and train the model
    model = create_model()
    model.fit(images, labels, epochs=10, batch_size=32)

    # Save the trained model
    model.save("red_green_model_with_percentages.h5")
    print("Model saved as red_green_model_with_percentages.h5")

    return model


# Run the training
train_model(r"I:\20 nov- Without model code\Images")
