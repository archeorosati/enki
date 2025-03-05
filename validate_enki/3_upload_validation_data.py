import os
import urllib.request
import zipfile

# ✅ Define the base directory for storing the dataset
base_dir = os.path.expanduser("~/Desktop/enki/enki_dataset") if os.name != "nt" else os.path.join(os.environ["USERPROFILE"], "Desktop", "enki", "enki_dataset")

# 📌 Define the URL and target directory
dataset_url = "https://zenodo.org/records/14950565/files/val_norm.zip"
dataset_zip = os.path.join(base_dir, "val_norm.zip")
dataset_extract_path = os.path.join(base_dir, "val_norm")

# ✅ Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

def download_and_extract():
    """ Downloads and extracts the val_norm dataset from Zenodo. """
    print(f"📥 Downloading dataset from {dataset_url}...")
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    
    os.remove(dataset_zip)
    print(f"✅ Dataset extracted successfully to {dataset_extract_path}")

if __name__ == "__main__":
    download_and_extract()
