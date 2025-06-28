import pandas as pd
import json

# CSV yolu
CSV_PATH = r"C:\Users\ferit\quality_control_project\data\annotations\jarlids_annots.csv"

# Dosyayı oku
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()

# JSON stringleri ayrıştırmak için yardımcı fonksiyon
def parse_json_column(json_str, field):
    try:
        data = json.loads(json_str)
        return data.get(field)
    except:
        return None

# Yeni sütunları oluştur: x, y, width, height
df['x'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'x'))
df['y'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'y'))
df['width'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'width'))
df['height'] = df['region_shape_attributes'].apply(lambda x: parse_json_column(x, 'height'))

# Etiket bilgisini çıkar (intact/damaged)
df['class_label'] = df['region_attributes'].apply(lambda x: parse_json_column(x, 'type'))

# Kontrol: İlk 5 satır
print("İlk 5 satır (işlenmiş):")
print(df[['filename', 'x', 'y', 'width', 'height', 'class_label']].head())
print("-" * 40)

# Sınıf dağılımı
print("Sınıf Dağılımı:")
print(df['class_label'].value_counts())
print("-" * 40)

# Toplam görsel sayısı
print(f"Toplam farklı görsel sayısı: {df['filename'].nunique()}")

# Görsel başına anotasyon sayısı (ilk 5)
print("\nGörsel başına anotasyon sayısı (örnek):")
print(df['filename'].value_counts().head())
