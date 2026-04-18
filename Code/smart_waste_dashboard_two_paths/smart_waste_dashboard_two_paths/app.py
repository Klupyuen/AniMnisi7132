from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime
import sqlite3
import cv2

# =========================
# CAMERA RTSP CONFIG
# IMPORTANT:
# 1. Replace these 3 values with your real camera details
# 2. If your teammate detection script is already using stream1,
#    use stream2 here for dashboard live view
# =========================
CAMERA_IP = "192.168.0.2"
CAMERA_USERNAME = "Paratiro"
CAMERA_PASSWORD = "ParaTiro321"

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/stream2"

app = Flask(__name__)

API_KEY = "SMART_WASTE_2026"
DB_PATH = "database.db"

latest_camera = {
    "type": None,
    "object_name": None,
    "timestamp": None,
    "image_base64": None
}

latest_system = {
    "weights": {
        "plastic": 0.0,
        "glass": 0.0,
        "metal": 0.0,
        "paper": 0.0
    },
    "bin_levels": {
        "plastic": "NOT FULL",
        "glass": "NOT FULL",
        "metal": "NOT FULL",
        "paper": "NOT FULL"
    },
    "emergency_stop": False,
    "motor_status": "IDLE",
    "timestamp": None
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS camera_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            object_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_base64 TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plastic_weight REAL NOT NULL,
            glass_weight REAL NOT NULL,
            metal_weight REAL NOT NULL,
            paper_weight REAL NOT NULL,
            plastic_status TEXT NOT NULL,
            glass_status TEXT NOT NULL,
            metal_status TEXT NOT NULL,
            paper_status TEXT NOT NULL,
            emergency_stop INTEGER NOT NULL,
            motor_status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def check_api_key():
    return request.headers.get("x-api-key") == API_KEY


def normalize_status(value):
    text = str(value).strip().lower()
    if text in {"full", "1", "true", "yes"}:
        return "FULL"
    return "NOT FULL"


def camera_row_to_dict(row):
    return {
        "id": row[0],
        "type": row[1],
        "object_name": row[2],
        "timestamp": row[3],
        "image_base64": row[4]
    }


def system_row_to_dict(row):
    return {
        "id": row[0],
        "weights": {
            "plastic": row[1],
            "glass": row[2],
            "metal": row[3],
            "paper": row[4]
        },
        "bin_levels": {
            "plastic": row[5],
            "glass": row[6],
            "metal": row[7],
            "paper": row[8]
        },
        "emergency_stop": bool(row[9]),
        "motor_status": row[10],
        "timestamp": row[11]
    }


def get_kpi_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_states ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "heaviest_compartment": "N/A",
            "heaviest_weight": 0,
            "full_bins": 0,
            "motor_status": "IDLE"
        }

    latest = system_row_to_dict(row)
    weights = latest["weights"]
    heaviest_compartment = max(weights, key=weights.get)
    heaviest_weight = weights[heaviest_compartment]
    full_bins = sum(1 for v in latest["bin_levels"].values() if v == "FULL")

    return {
        "heaviest_compartment": heaviest_compartment,
        "heaviest_weight": heaviest_weight,
        "full_bins": full_bins,
        "motor_status": latest["motor_status"]
    }


# =========================
# LIVE CAMERA STREAM FOR DASHBOARD
# =========================
def generate_live_frames():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open RTSP stream for dashboard.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_live_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM camera_events ORDER BY id DESC LIMIT 10")
    camera_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM system_states ORDER BY id DESC LIMIT 10")
    system_rows = cursor.fetchall()

    conn.close()

    camera_history = [camera_row_to_dict(r) for r in camera_rows]
    system_history = [system_row_to_dict(r) for r in system_rows]

    latest_cam = camera_history[0] if camera_history else latest_camera
    latest_sys = system_history[0] if system_history else latest_system
    chart_labels = [entry["timestamp"] for entry in reversed(system_history)]
    chart_values = [sum(entry["weights"].values()) for entry in reversed(system_history)]

    return render_template(
        "dashboard.html",
        latest_cam=latest_cam,
        latest_sys=latest_sys,
        camera_history=camera_history,
        system_history=system_history,
        chart_labels=chart_labels,
        chart_values=chart_values,
        analytics=get_kpi_summary()
    )


@app.route("/health")
def health():
    return jsonify({"status": "server running"})


@app.route("/version")
def version():
    return jsonify({"system": "Smart Waste Classification Bin", "version": "4.1"})


@app.route("/camera_event", methods=["POST"])
def camera_event():
    global latest_camera

    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    required_fields = ["type", "object_name"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    timestamp = data.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    image_b64 = data.get("image_base64")

    latest_camera = {
        "type": str(data["type"]).strip().lower(),
        "object_name": str(data["object_name"]).strip().lower(),
        "timestamp": timestamp,
        "image_base64": image_b64
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO camera_events (type, object_name, timestamp, image_base64) VALUES (?, ?, ?, ?)",
        (
            latest_camera["type"],
            latest_camera["object_name"],
            latest_camera["timestamp"],
            latest_camera["image_base64"]
        )
    )
    conn.commit()
    conn.close()

    print("Camera event received:", latest_camera, flush=True)
    return jsonify({"message": "Camera event received successfully"}), 200


@app.route("/system_state", methods=["POST"])
def system_state():
    global latest_system

    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    required_fields = ["weights", "bin_levels"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    keys = ["plastic", "glass", "metal", "paper"]
    for key in keys:
        if key not in data["weights"]:
            return jsonify({"error": f"Missing weights.{key}"}), 400
        if key not in data["bin_levels"]:
            return jsonify({"error": f"Missing bin_levels.{key}"}), 400

    timestamp = data.get("timestamp") or datetime.now().isoformat(timespec="seconds")

    latest_system = {
        "weights": {
            "plastic": float(data["weights"]["plastic"]),
            "glass": float(data["weights"]["glass"]),
            "metal": float(data["weights"]["metal"]),
            "paper": float(data["weights"]["paper"])
        },
        "bin_levels": {
            "plastic": normalize_status(data["bin_levels"]["plastic"]),
            "glass": normalize_status(data["bin_levels"]["glass"]),
            "metal": normalize_status(data["bin_levels"]["metal"]),
            "paper": normalize_status(data["bin_levels"]["paper"])
        },
        "emergency_stop": bool(data.get("emergency_stop", False)),
        "motor_status": str(data.get("motor_status", "IDLE")).strip().upper(),
        "timestamp": timestamp
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO system_states (
            plastic_weight, glass_weight, metal_weight, paper_weight,
            plastic_status, glass_status, metal_status, paper_status,
            emergency_stop, motor_status, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            latest_system["weights"]["plastic"],
            latest_system["weights"]["glass"],
            latest_system["weights"]["metal"],
            latest_system["weights"]["paper"],
            latest_system["bin_levels"]["plastic"],
            latest_system["bin_levels"]["glass"],
            latest_system["bin_levels"]["metal"],
            latest_system["bin_levels"]["paper"],
            1 if latest_system["emergency_stop"] else 0,
            latest_system["motor_status"],
            latest_system["timestamp"]
        )
    )
    conn.commit()
    conn.close()

    print("System state received:", latest_system, flush=True)
    return jsonify({"message": "System state received successfully"}), 200


@app.route("/latest_camera")
def latest_camera_route():
    return jsonify(latest_camera)


@app.route("/latest_system")
def latest_system_route():
    return jsonify(latest_system)


@app.route("/camera_history")
def camera_history_route():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM camera_events ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([camera_row_to_dict(r) for r in rows])


@app.route("/system_history")
def system_history_route():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_states ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([system_row_to_dict(r) for r in rows])


if __name__ == "__main__":
    init_db()
    app.run(debug=True)