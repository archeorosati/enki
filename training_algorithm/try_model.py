try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.applications import EfficientNetB0, ResNet50
except ModuleNotFoundError:
    raise ImportError("TensorFlow is not installed. Please install it using 'pip install tensorflow' before running this script.")

def build_model(base_model_name="EfficientNetB0", img_height=350, img_width=350, num_classes=2, freeze_layers=75):
    """
    Builds the deep learning model based on the selected base architecture.
    """
    if base_model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    else:
        raise ValueError("Unsupported base model. Choose either 'EfficientNetB0' or 'ResNet50'")

    # Freeze initial layers
    base_model.trainable = True
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Define custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax")  # Multi-class classification
    ])

    return model
