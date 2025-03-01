import tensorflow as tf
import os
import urllib.request
import zipfile

def download_and_extract_dataset(url, extract_to):
    """
    Downloads and extracts dataset from the given Zenodo URL.
    """
    dataset_zip = os.path.join(extract_to, "dataset.zip")
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f"📥 Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, dataset_zip)
    
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(dataset_zip)
    print("✅ Dataset ready.")

def load_dataset(data_dir, img_height=350, img_width=350, batch_size=128, validation_split=0.2):
    """
    Loads dataset from the given directory and prepares it for training.
    The dataset is expected to have 'positive' and 'negative' folders for classification.
    If the dataset is not found locally, it will be downloaded from Zenodo.
    """
    zenodo_url = "https://zenodo.org/records/14950565/files/Normalized_Tell_combined_350.zip"
    
    # Define the target directory on Desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Enki", "enki_dataset")
    data_dir = os.path.join(desktop_path, "Normalized_Tell_combined_350")
    
    if not os.path.exists(data_dir):
        print("⚠️ Dataset not found locally. Downloading from Zenodo...")
        download_and_extract_dataset(zenodo_url, desktop_path)
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"📂 Classes found: {class_names} - Number of classes: {num_classes}")

    # Apply prefetching to speed up training
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, class_names, num_classes
