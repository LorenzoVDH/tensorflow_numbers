import tensorflow as tf
from PIL import Image
import numpy as np
import sys 
import os

# Load the saved model
model = tf.keras.models.load_model('digit_recognition_model.keras')
print("Model loaded from 'digit_recognition_model.keras'")

# Now you can use it for prediction, like in the previous step
# (Assuming you already have the image processing code from before)

# Example: Predict the digit for a custom image
# Get the image path from command line arguments
image_path = sys.argv[1] if len(sys.argv) > 1 else "imagetest.png"

# Check if the provided path is valid
if not os.path.isfile(image_path):
    print(f"Warning: The provided image path '{image_path}' is not valid. Using default image 'imagetest.png'.")
    image_path = "imagetest.png"

print("Using image path:", image_path)
img = Image.open(image_path)

# Convert, resize, normalize, and reshape the image as shown earlier
img = img.convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28
img_array = np.array(img) / 255.0  # Normalize
img_array = img_array.reshape((1, 28, 28, 1))  # Reshape to match model input shape

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print(f'Predicted digit: {predicted_digit}')

