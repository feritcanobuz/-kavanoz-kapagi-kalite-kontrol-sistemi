import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, roc_auc_score,
    mean_squared_error
)

# === Parametreler ===
DATA_DIR = r"C:\Users\ferit\quality_control_project\data\split"
MODEL_PATH = os.path.join(DATA_DIR, "model_final_cnn.keras")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# === Validation seti yükle ===
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int",
    shuffle=False
)

# === Normalizasyon ===
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# === Modeli yükle ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Tahminler ===
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_probs = model.predict(val_ds).flatten()
y_pred = (y_probs > 0.48).astype(int)

# === 1. Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
labels = val_ds.class_names if hasattr(val_ds, "class_names") else ['kusurlu', 'kusursuz']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# === 2. Classification Report ===
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=labels))

# === 3. Sigmoid Çıkışlarının Dağılımı ===
plt.figure()
plt.hist(y_probs[y_true == 0], bins=25, alpha=0.6, label="kusurlu (gerçek)")
plt.hist(y_probs[y_true == 1], bins=25, alpha=0.6, label="kusursuz (gerçek)")
plt.title("Model Güveni Dağılımı")
plt.xlabel("Sigmoid Çıkışı (Model Güveni)")
plt.ylabel("Örnek Sayısı")
plt.legend()
plt.show()

# === 4. ROC Eğrisi ve AUC ===
fpr, tpr, _ = roc_curve(y_true, y_probs)
auc_score = roc_auc_score(y_true, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Eğrisi")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# === 5. MSE & RMSE ===
mse = mean_squared_error(y_true, y_probs)
rmse = np.sqrt(mse)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AUC : {auc_score:.4f}")

