import base64
import requests
from pathlib import Path
from datetime import datetime

url = "http://127.0.0.1:5000/camera_event"
headers = {"x-api-key": "SMART_WASTE_2026"}

image_path = Path("sample.jpg")
image_b64 = None
if image_path.exists():
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

data = {
    "type": "plastic",
    "object_name": "plastic bottle",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "image_base64": image_b64
}

response = requests.post(url, json=data, headers=headers)
print("Status code:", response.status_code)
print("Response:", response.text)
