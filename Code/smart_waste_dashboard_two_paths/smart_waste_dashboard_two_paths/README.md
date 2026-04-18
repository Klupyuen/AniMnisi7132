# Smart Waste Classification Bin - Two-Path Version

This version implements your updated design.

## Path 1: Camera
Camera / Groq vision code -> HTTP JSON -> Flask `/camera_event` -> database -> dashboard

## Path 2: Sensors / ESP32
Sensors + emergency stop + motor -> ESP32 -> HTTP JSON -> Flask `/system_state` -> database -> dashboard

The dashboard shows:
- detected type
- object name
- latest camera image if `image_base64` is provided
- latest system state
- recent history
- analytics summary

## Install required libraries

`py -m pip install flask requests pandas numpy matplotlib`

If you want to use the provided camera script:

`py -m pip install opencv-python groq`

## Main files

- `app.py`
- `init_db.py`
- `check_db.py`
- `scripts/send_camera_event.py`
- `scripts/send_system_state.py`
- `scripts/send_fake_data.py`
- `scripts/camera_to_flask.py`
- `scripts/analytics_basic.py`
- `scripts/generate_report.py`
- `templates/dashboard.html`

## Run order

1. `py init_db.py`
2. `py -m flask --app app run --no-reload`
3. In another terminal:
   - `py scripts/send_camera_event.py`
   - `py scripts/send_system_state.py`
4. Open dashboard:
   - `http://127.0.0.1:5000`
