import os
import urllib.request
import zipfile

# Set directories
DEST_DIR = os.path.expanduser("~/Desktop/enki/enki_dataset")
DATASET_URL = "https://zenodo.org/records/14950565/files/Normalized_Tell_combined_350.zip"

def download_and_extract_dataset():
    """
    Downloads and extracts the dataset from the specified URL into the destination folder.
    """
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    dataset_zip = os.path.join(DEST_DIR, "Normalized_Tell_combined_350.zip")
    
    print(f"📥 Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, dataset_zip)
    
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
    
    os.remove(dataset_zip)
    print("✅ Dataset ready in", DEST_DIR)

if __name__ == "__main__":
    download_and_extract_dataset()
    print("🎉 Download and extraction completed!")
