import os
import random
import shutil

# Klasör yolları
train_dir = r"C:\Users\ferit\quality_control_project\data\split\train"
test_dir  = r"C:\Users\ferit\quality_control_project\data\split\test"

classes = ['kusurlu', 'kusursuz']
test_ratio = 0.10  # %10 test için ayrılacak

for cls in classes:
    src_folder = os.path.join(train_dir, cls)
    dst_folder = os.path.join(test_dir, cls)

    images = os.listdir(src_folder)
    selected = random.sample(images, int(len(images) * test_ratio))

    for img_name in selected:
        src = os.path.join(src_folder, img_name)
        dst = os.path.join(dst_folder, img_name)
        shutil.copy2(src, dst)  # dosyayı kopyala
