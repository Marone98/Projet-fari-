import base64
import requests

IMAGE_PATH = "tests/images/test.png"
URL = "http://127.0.0.1:5000/play"

with open(IMAGE_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": "data:image/png;base64," + img_b64
}

r = requests.post(URL, json=payload)
print("Status:", r.status_code)
print("Response:", r.text)
