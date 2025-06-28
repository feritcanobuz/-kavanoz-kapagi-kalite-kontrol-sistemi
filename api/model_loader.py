from keras.models import load_model
import numpy as np
import cv2
import os

# === MODELİ YÜKLE ===
# Bu yol, api klasöründen çalıştırıldığında doğrudur
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join("data", "split", "model_final_cnn.keras")

print(" Model Yolu:", MODEL_PATH)

model = load_model(MODEL_PATH)
print(" Model yüklendi:", model.input_shape)


# === GÖRSEL ÖN İŞLEME ===
def preprocess_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Görsel okunamadı. Desteklenmeyen format olabilir.")

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (128,128) → (128,128,1)
    img = np.expand_dims(img, axis=0)   # (128,128,1) → (1,128,128,1)

    print(" Input shape:", img.shape)
    return img


# === TAHMİN ===
def load_model_and_predict(image_bytes):
    try:
        img = preprocess_image(image_bytes)
        prob = model.predict(img, verbose=0)[0][0]
        label = "kusursuz" if prob >= 0.5 else "kusurlu"
        print(f" Tahmin: {label} ({prob:.4f})")
        return label, prob
    except Exception as e:
        print(" Tahmin hatası:", e)
        return "tahmin edilemedi", 0.0
