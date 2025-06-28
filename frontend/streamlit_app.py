import streamlit as st
import requests
import io
from PIL import Image
from collections import Counter
from datetime import datetime

st.set_page_config(page_title="Kalite Kontrol", layout="wide")

st.title("🔍 Kalite Kontrol Görsel Sınıflandırma")

st.markdown("""
Bu uygulama, yüklenen kavanoz kapağı görsellerini inceleyerek **kusurlu** ya da **kusursuz** olarak sınıflandırır.  
Model, derin öğrenme tabanlı bir CNN mimarisi kullanılarak eğitilmiştir.  
API üzerinden tahminler yapılmakta ve sonuçlar aşağıda detaylı biçimde sunulmaktadır.
""")

with st.sidebar:
    st.markdown("### 📌 Model Bilgisi")
    st.info("""
    **Model:** CNN (model_final_cnn.keras)
    **Aktivasyon Fonksiyonu:** Sigmoid  
    **Doğruluk:** %83  
    **Giriş boyutu:** 128x128  
    **Tahmin API:** http://quality_api:8000/predict
    """)

uploaded_files = st.file_uploader("📤 Görselleri seçin", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    predictions = []
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    for index, file in enumerate(uploaded_files, start=1):
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        files = {'files': (file.name, image_bytes, 'image/png')}

        try:
            response = requests.post("http://quality_api:8000/predict", files=files)
            if response.ok:
                result = response.json()["results"][0]
                predictions.append({
                    "filename": file.name,
                    "image": image,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "timestamp": now,
                    "index": index
                })
            else:
                st.error(f"{file.name} tahmin edilemedi.")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

    # 🔢 Sınıf Sayımı
    st.markdown("## 🔢 Sınıf Sayımı")
    class_counts = Counter([p["prediction"] for p in predictions])
    for label, count in class_counts.items():
        color = {"kusursuz": "green", "kusurlu": "red"}.get(label, "gray")
        st.markdown(f"- <span style='color:{color}; font-weight:bold'>{label}</span>: {count} adet", unsafe_allow_html=True)

    # 📋 Tahmin Sonuçları
    st.markdown("---")
    st.markdown("## 📋 Tahmin Sonuçları")

    for i in range(0, len(predictions), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(predictions):
                p = predictions[i + j]
                with cols[j]:
                    with st.container():
                        st.markdown(f"#### 📦 Görsel {p['index']}: {p['filename']}")
                        
                        # 🎯 Görsel sabit boyutta gösterilsin
                        st.image(p["image"], width=400)

                        label_color = "green" if p["prediction"] == "kusursuz" else "red"
                        st.markdown(
                            f"**🔍 Sonuç:** <span style='color:{label_color}; font-weight:bold'>{p['prediction']}</span>",
                            unsafe_allow_html=True
                        )

                        confidence_percent = p["confidence"] * 100
                        confidence_color = (
                            "orange" if 40 <= confidence_percent <= 60 else
                            ("green" if p["prediction"] == "kusursuz" else "red")
                        )

                        st.markdown(
                            f"**📊 Güven:** <span style='color:{confidence_color}'>{confidence_percent:.2f}%</span>",
                            unsafe_allow_html=True
                        )

                        if 0.4 <= p["confidence"] <= 0.6:
                            st.warning("⚠️ Model bu görselde kararsız (emin değil).")

                        st.markdown(f"🕒 Tahmin zamanı: `{p['timestamp']}`")

    # Model özeti
    st.markdown("---")
    st.markdown("### 📈 Model Performansı (Özet)")
    st.success("✔️ Model başarıyla yüklendi. Ortalama doğruluk: **%83**")
