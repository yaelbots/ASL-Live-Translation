import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/model1")

# Function to load and preprocess test images
def load_and_preprocess_test_data(test_data_dir, target_size=(150, 150)):
    test_images = []
    for image_name in os.listdir(test_data_dir):
        image_path = os.path.join(test_data_dir, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = image / 255.0  # Normalize pixel values
            test_images.append(image)
    return np.array(test_images)

# Directory containing test images
test_data_dir = "data/asl_train/test_model"

# Load and preprocess test data
test_images = load_and_preprocess_test_data(test_data_dir)

# Reshape images to add channel dimension
test_images = np.expand_dims(test_images, axis=-1)

# Make predictions on test images
predictions = model.predict(test_images)

# Convert predictions to class labels (assuming one-hot encoding)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to letters (if needed)
predicted_letters = [chr(ord('A') + class_idx) for class_idx in predicted_classes]

# Print the predicted letters
print("Predicted letters:", predicted_letters)

