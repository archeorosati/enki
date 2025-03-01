import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50  # ✅ Added alternative model for GPU optimization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ✅ Automatically detect GPU and configure TensorFlow accordingly
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth on available GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Running on GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU")

# 📂📂📂⚠️ Pay attention: Define directories using relative paths
data_dir = "../data/Normalized_Tell_combined_350"
output_dir = "../models"
report_dir = "../reports"

# 📌 Configuration parameters
img_height, img_width = 350, 350  # ✅ Reduced size for better memory management
batch_size = 128  # ✅ Lower batch size to avoid OOM errors on Mac
num_epochs = 10

# ✅ Efficient dataset loading
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# ✅ Extract class labels
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"📂 Classes found: {class_names} - Number of classes: {num_classes}")

# ✅ Apply prefetching for faster training
train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# 📌 Choose model based on platform optimization
use_mac_optimized = False  # Set to True for Mac M1/M2

if use_mac_optimized:
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
else:
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))

base_model.trainable = False  # Freeze initial weights

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dense(num_classes, activation="softmax")  # ✅ Softmax for multi-class classification
])

# 📌 Compile the model with Adam optimizer and a reduced learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 📌 Define callbacks for training stability
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(os.path.join(output_dir, "best_model.keras"), save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

# 🚀 Optimized training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# ✅ Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:75]:  # Freeze only the first 75 layers
    layer.trainable = False

# 📌 Recompile with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 🚀 Fine-tuning for another 10 epochs
history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# 📌 Save the final model
model.save(os.path.join(output_dir, "model_final.keras"))

# 📌 Generate Training Report

def generate_report(history, test_acc, report_dir):
    epochs = np.arange(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    graph_path = os.path.join(report_dir, "training_report.png")
    plt.savefig(graph_path)
    plt.close()

    # 📌 Create PDF report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = os.path.join(report_dir, f"Report_{timestamp}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Training Report", style=styles["Title"]))
    elements.append(Paragraph(f"Test Accuracy: {test_acc:.2f}", style=styles["Heading2"]))
    elements.append(Spacer(1, 12))

    table = Table([["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"]] +
                  list(zip(epochs, history.history["accuracy"], history.history["val_accuracy"],
                           history.history["loss"], history.history["val_loss"])))
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 1, colors.black)]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Image(graph_path, width=6 * inch, height=3 * inch))
    doc.build(elements)
    print(f"📄 Report saved in {pdf_path}")

# 📌 Generate the final report
test_loss, test_acc = model.evaluate(val_dataset)
generate_report(history, test_acc, report_dir)
