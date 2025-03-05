import os
import numpy as np
from PIL import Image

# ✅ Automatically detect OS and set paths accordingly
base_dir = os.path.expanduser("~/Desktop/enki/enki_dataset") if os.name != "nt" else os.path.join(os.environ["USERPROFILE"], "Desktop", "enki", "enki_dataset")

# 📌 Define relative dataset and output directories changing the following placeholder with your own images
dataset_path = os.path.join(base_dir, "Negative", "20241029")

# 📌 Define relative dataset and output directories changing the following placeholder
output_path = os.path.join(base_dir, "Normalized_Tell_combined_350", "Negative")

# ✅ Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# ✅ Function to load and preprocess an image
def load_and_preprocess_image(filepath, target_size=(64, 64)):
    try:
        img = Image.open(filepath)
        img = img.resize(target_size)  # Resize the image
        img_array = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
        return img_array
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None

# ✅ Function to save a processed image
def save_image(image_array, save_path):
    img = (image_array * 255).astype(np.uint8)  # Convert back to 0-255
    img = Image.fromarray(img)
    img.save(save_path)

# ✅ Process and save images
for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.png')):
        filepath = os.path.join(dataset_path, filename)
        img = load_and_preprocess_image(filepath)
        if img is not None:
            save_path = os.path.join(output_path, filename)
            save_image(img, save_path)

print(f"✅ All preprocessed images have been saved in {output_path}")
