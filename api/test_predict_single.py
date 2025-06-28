
import requests

with open("data/split/train/kusurlu/p1_0.png", "rb") as f:
    response = requests.post("http://127.0.0.1:8000/predict", files={"files": f})

print(response.json())
