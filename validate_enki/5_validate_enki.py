import tensorflow as tf
import os
import pandas as pd
from pathlib import Path
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory

# ✅ Enable GPU Metal support on macOS for performance optimization
os.environ["TF_METAL"] = "1"
tf.config.list_physical_devices("GPU")

# 📂 Define the absolute path for the dataset directory
base_dir = os.path.expanduser("~/Desktop/enki/enki_dataset") if os.name != "nt" else os.path.join(os.environ["USERPROFILE"], "Desktop", "enki", "enki_dataset")

# 📌 Directories for dataset, models, and reports
data_dir = base_dir  
output_dir = Path(base_dir) / "models"
report_dir = Path(base_dir) / "reports"
model_dir = Path(base_dir).parent / "models"
model_path = model_dir / "enki.keras"
val_dir = Path(base_dir) / "validation"  # Define validation dataset directory

# 📌 Ensure necessary directories exist
output_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

# ✅ Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ✅ Optimize validation dataset
batch_size = 64  # 🔥 Use 64 instead of 128 to reduce memory consumption

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=(350, 350),
    batch_size=batch_size,
    shuffle=False
)

# ✅ Optimize with cache & prefetch
val_dataset = val_dataset.cache().prefetch(AUTOTUNE)

# 🚀 Model evaluation
num_batches = tf.data.experimental.cardinality(val_dataset).numpy()
val_loss = 0
val_accuracy = 0
total_samples = 0

for batch_images, batch_labels in val_dataset:
    batch_loss, batch_acc = model.evaluate(batch_images, batch_labels, verbose=0)
    batch_size_actual = batch_labels.shape[0]
    val_loss += batch_loss * batch_size_actual
    val_accuracy += batch_acc * batch_size_actual
    total_samples += batch_size_actual

val_loss /= total_samples
val_accuracy /= total_samples

# 📌 Save results to CSV
results = {
    "Validation Loss": [val_loss],
    "Validation Accuracy": [val_accuracy],
}

results_df = pd.DataFrame(results)
csv_path = output_dir / "validation_results_optimized.csv"
results_df.to_csv(csv_path, index=False)

print(f"✅ Validation complete. Results saved to: {csv_path}")
print(f"📊 Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
