import requests
import random
import time
from datetime import datetime

camera_url = "http://127.0.0.1:5000/camera_event"
system_url = "http://127.0.0.1:5000/system_state"
headers = {"x-api-key": "SMART_WASTE_2026"}

types = ["plastic", "metal", "paper", "glass"]
objects = {
    "plastic": ["plastic bottle", "plastic cup", "wrapper"],
    "metal": ["metal can", "metal lid", "metal ball"],
    "paper": ["paper sheet", "box", "paper cup"],
    "glass": ["glass bottle", "glass cup", "glass jar"]
}

base_weights = {"plastic": 100, "glass": 150, "metal": 120, "paper": 90}

for i in range(20):
    detected_type = random.choice(types)
    camera_payload = {
        "type": detected_type,
        "object_name": random.choice(objects[detected_type]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_base64": None
    }
    cam_resp = requests.post(camera_url, json=camera_payload, headers=headers)

    for key in base_weights:
        base_weights[key] += random.randint(1, 20)

    system_payload = {
        "weights": {
            "plastic": base_weights["plastic"],
            "glass": base_weights["glass"],
            "metal": base_weights["metal"],
            "paper": base_weights["paper"]
        },
        "bin_levels": {
            "plastic": "FULL" if base_weights["plastic"] > 320 else "NOT FULL",
            "glass": "FULL" if base_weights["glass"] > 320 else "NOT FULL",
            "metal": "FULL" if base_weights["metal"] > 320 else "NOT FULL",
            "paper": "FULL" if base_weights["paper"] > 320 else "NOT FULL"
        },
        "emergency_stop": random.choice([False, False, False, True]),
        "motor_status": random.choice(["RUNNING", "STOPPED", "IDLE"]),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    sys_resp = requests.post(system_url, json=system_payload, headers=headers)

    print(f"Cycle {i+1}: camera={cam_resp.status_code}, system={sys_resp.status_code}")
    time.sleep(1)
