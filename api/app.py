from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_model_and_predict  # Bu senin utils fonksiyonun
from typing import List
import uvicorn

app = FastAPI()

# CORS (Frontend ile konuşmak için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        prediction, confidence = load_model_and_predict(contents)
        results.append({
            "filename": file.filename,
            "prediction": prediction,  # "kusurlu" / "kusursuz"
            "confidence": float(confidence)
        })

    return {"results": results}

# Eğer script olarak çalıştırılırsa
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
