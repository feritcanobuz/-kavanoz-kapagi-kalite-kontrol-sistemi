import os
import pandas as pd
from PIL import Image
import json

# Ana klasör yolu
BASE_DIR = r"C:\Users\ferit\quality_control_project"

# Giriş dosyalarının yolları
CSV_PATH = os.path.join(BASE_DIR, "data", "annotations", "jarlids_annots.csv")
IMG_DIR = os.path.join(BASE_DIR, "data", "raw")

# Çıkış klasörleri
OUTPUT_DIR_DAMAGED = os.path.join(BASE_DIR, "data", "dataset", "kusurlu")
OUTPUT_DIR_INTACT = os.path.join(BASE_DIR, "data", "dataset", "kusursuz")

# Etiket haritası
LABEL_MAP = {
    'damaged': OUTPUT_DIR_DAMAGED,
    'intact': OUTPUT_DIR_INTACT
}

# CSV dosyasını oku ve sütunları temizle
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()

# JSON string içeren sütunları ayrıştıran fonksiyon
def parse_json_column(json_str, field):
    try:
        data = json.loads(json_str)
        return data.get(field)
    except Exception:
        return None

# Koordinat ve etiket bilgilerini ayrıştır
df['x'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'x'))
df['y'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'y'))
df['width'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'width'))
df['height'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'height'))
df['class_label'] = df['region_attributes'].apply(lambda x: parse_json_column(x, 'type'))

# Her satırı işle: kırp ve etiketle
for idx, row in df.iterrows():
    filename = row['filename']
    label = str(row['class_label']).strip().lower()
    x, y, w, h = row['x'], row['y'], row['width'], row['height']

    # Eksik veri varsa atla
    if None in [filename, label, x, y, w, h]:
        print(f"Geçersiz veri atlandı (satır {idx})")
        continue

    img_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(img_path):
        print(f"Görsel bulunamadı: {img_path}")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))

        output_folder = LABEL_MAP.get(label)
        if output_folder is None:
            print(f"Bilinmeyen etiket: {label} (satır {idx})")
            continue

        save_name = f"{os.path.splitext(filename)[0]}_{idx}.png"
        save_path = os.path.join(output_folder, save_name)
        crop.save(save_path)

    except Exception as e:
        print(f"Hata oluştu ({filename}, satır {idx}): {e}")

print("Kırpma ve etiketleme tamamlandı.")
