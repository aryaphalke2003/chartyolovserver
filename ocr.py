import os
import pytesseract
from PIL import Image

# Path to the crops folder
crops_folder = 'runs/detect/exp7/crops'

# Initialize the OCR engine
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Update with your Tesseract executable path

# Function to apply OCR to an image
def apply_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print(text)
    return text.strip()

# Initialize the dictionary to store results
results_dict = {}

# Iterate through each folder in crops folder
for folder_name in os.listdir(crops_folder):
    folder_path = os.path.join(crops_folder, folder_name)
    if os.path.isdir(folder_path):
        texts = []
        # Iterate through each image in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                # Apply OCR to the image
                text = apply_ocr(image_path)
                texts.append(text)
        # Store the texts in the results dictionary
        results_dict[folder_name] = texts

# Print the results dictionary
for key, value in results_dict.items():
    print(f"{key} - {value}")
