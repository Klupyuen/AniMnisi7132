import requests
from datetime import datetime

url = "http://127.0.0.1:5000/system_state"
headers = {"x-api-key": "SMART_WASTE_2026"}

data = {
    "weights": {
        "plastic": 120,
        "glass": 400,
        "metal": 250,
        "paper": 180
    },
    "bin_levels": {
        "plastic": "NOT FULL",
        "glass": "FULL",
        "metal": "NOT FULL",
        "paper": "NOT FULL"
    },
    "emergency_stop": False,
    "motor_status": "RUNNING",
    "timestamp": datetime.now().isoformat(timespec="seconds")
}

response = requests.post(url, json=data, headers=headers)
print("Status code:", response.status_code)
print("Response:", response.text)
