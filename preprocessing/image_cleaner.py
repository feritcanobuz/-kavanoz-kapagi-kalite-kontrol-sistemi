import os
import cv2

# Giriş ve çıkış klasörlerini tanımla
BASE_DIR = r"C:\Users\ferit\quality_control_project"
INPUT_DIR = os.path.join(BASE_DIR, "data", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

# Hedef boyut (CNN girişi için sabit)
TARGET_SIZE = (128, 128)

# Klasörleri sırayla işle
for label in ["kusurlu", "kusursuz"]:
    input_folder = os.path.join(INPUT_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    # Çıkış klasörünü oluştur (varsa geç)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, filename)

        try:
            # Görseli oku
            img = cv2.imread(img_path)

            # Gri tonlamaya çevir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Gaussian blur uygula
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Yeniden boyutlandır
            resized = cv2.resize(blurred, TARGET_SIZE)

            # Tek kanal (grayscale) olarak kaydet
            cv2.imwrite(save_path, resized)

        except Exception as e:
            print(f"Hata oluştu: {filename} → {e}")

print("Ön işleme tamamlandı. İşlenmiş görseller 'preprocessed/' klasörüne kaydedildi.")
