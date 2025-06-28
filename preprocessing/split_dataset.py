import os
import shutil
import random

# Ana klasörler
BASE_DIR = r"C:\Users\ferit\quality_control_project\data"
INPUT_DIR = os.path.join(BASE_DIR, "preprocessed")
OUTPUT_DIR = os.path.join(BASE_DIR, "split")

# Eğitim/validation oranı
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Hangi sınıflar var
labels = ["kusurlu", "kusursuz"]

for label in labels:
    input_folder = os.path.join(INPUT_DIR, label)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Karıştır ve böl
    random.shuffle(files)
    train_count = int(len(files) * TRAIN_RATIO)
    train_files = files[:train_count]
    val_files = files[train_count:]

    # Hedef klasörleri oluştur
    for split_type, split_files in [("train", train_files), ("val", val_files)]:
        target_dir = os.path.join(OUTPUT_DIR, split_type, label)
        os.makedirs(target_dir, exist_ok=True)

        for f in split_files:
            src = os.path.join(input_folder, f)
            dst = os.path.join(target_dir, f)
            shutil.copy2(src, dst)

    print(f"{label}: {len(train_files)} train / {len(val_files)} val")

print(" Veri başarıyla split/train ve split/val klasörlerine ayrıldı.")


