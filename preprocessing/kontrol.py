import os

base_path = r"C:\Users\ferit\quality_control_project\data\dataset"

kusurlu = os.listdir(os.path.join(base_path, "kusurlu"))
kusursuz = os.listdir(os.path.join(base_path, "kusursuz"))

print(f"Kusurlu sayısı  : {len(kusurlu)}")
print(f"Kusursuz sayısı : {len(kusursuz)}")
print(f"Toplam          : {len(kusurlu) + len(kusursuz)}")
