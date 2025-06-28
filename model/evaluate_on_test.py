import tensorflow as tf
import os

# === PARAMETRELER ===
DATA_DIR = r"C:\Users\ferit\quality_control_project\data\split"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# === TEST SETİNİ YÜKLEME ===
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int",
    shuffle=False
).map(lambda x, y: (x / 255.0, y))

# === MODELİ YÜKLE ===
from tensorflow.keras.models import load_model
model = load_model(os.path.join(DATA_DIR, "model_final_cnn.keras"))

# === TEST SKORLARI ===
loss, acc = model.evaluate(test_ds)
print(f"\n🎯 Test Loss: {loss:.4f} — Test Accuracy: {acc:.4f}")
