import cv2
import json
import base64
import time
import threading
import numpy as np
import requests
from groq import Groq
from ultralytics import YOLO

# =========================
# Configuration
# =========================
GROQ_API_KEY =
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

CAMERA_IP = "192.168.0.2"
CAMERA_USERNAME = "Paratiro"
CAMERA_PASSWORD = "ParaTiro321"

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/stream1"

# Flask endpoint on the SAME laptop
SERVER_URL = "http://127.0.0.1:5000/camera_event"
API_KEY = "SMART_WASTE_2026"

# YOLO presence gate only
YOLO_MODEL_PATH = r"C:\Users\ROG STRIX\OneDrive - Universiti Tunku Abdul Rahman\Digital Integrated Circuit Design\Desktop\UTAR\Y4S2\Embedded\yolov8x-worldv2.pt"
YOLO_CONF_THRESHOLD = 0.20
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMGSZ = 960
YOLO_MIN_BOX_AREA_RATIO = 0.003
YOLO_CROP_PADDING = 12

# Stability / state machine
DETECTION_HOLD_FRAMES = 10
REQUIRED_PRESENT_FRAMES = 4
REQUIRED_EMPTY_FRAMES = 10
DETECTION_DELAY_SECONDS = 1.0
SEND_COOLDOWN_SECONDS = 2.0

ALLOWED_MATERIALS = {"paper", "glass", "metal", "plastic", "others"}

ROI_X1_RATIO = 0.12
ROI_Y1_RATIO = 0.50
ROI_X2_RATIO = 0.92
ROI_Y2_RATIO = 0.98

TARGET_WIDTH = 640
JPEG_QUALITY = 55
MIN_VLM_WIDTH = 512

IGNORE_OBJECT_KEYWORDS = {
    "trash bin",
    "dustbin",
    "bin",
    "garbage bin",
    "rubbish bin",
    "waste bin",
    "container",
    "lid",
    "cover",
    "cap",
    "rim",
    "edge",
    "opening",
    "mouth",
    "handle",
    "hinge",
    "wall",
    "inner wall",
    "outer wall"
}

# Optional: reject obvious false triggers from the full scene
YOLO_REJECT_CLASSES = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "keyboard",
    "mouse",
    "toilet",
    "sink",
    "refrigerator",
    "oven",
    "microwave",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "airplane",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}

SYSTEM_PROMPT = """
You are a strict visual classifier.

Task:
The image has already been cropped to show only the target object region.

Important:
Ignore the trash bin itself and any part of the trash bin structure.

You must ignore:
- trash bin
- dustbin
- garbage bin
- bin lid
- cover
- rim
- edge
- opening
- handle
- hinge
- inner wall
- outer wall
- any structural part of the container

Your goal:
Identify only the main loose object that is NOT part of the trash bin.

Allowed materials:
paper
glass
metal
plastic
others

Rules:
1. Never return the trash bin or any part of it.
2. Only return a loose object that is separate from the bin structure.
3. Ignore anything that forms the boundary or structure of the bin.
4. Choose only one dominant material.
5. The material must be exactly one of:
   paper, glass, metal, plastic, others
6. If there is no valid loose object, return:
   {
     "object": "none",
     "material": "others"
   }
7. Return JSON only.
8. Use exactly these keys:
   object
   material
9. Do not include explanations, notes, markdown, or extra keys.
10. Keep the object name short, like:
   bottle, cup, box, bag, can, paper, wrapper
"""

USER_PROMPT = """
Identify the main loose object in this cropped image.

Ignore the trash bin, lid, rim, edge, cover, and all container structure.

Return only valid JSON in exactly this format:
{
  "object": "object_name_or_none",
  "material": "paper|glass|metal|plastic|others"
}
"""

# =========================
# Shared State
# =========================
latest_frame = None
frame_lock = threading.Lock()

latest_result = None
result_lock = threading.Lock()

stop_event = threading.Event()
vlm_busy = False
vlm_lock = threading.Lock()

last_send_time = 0.0

# Presence gate state
present_counter = 0
empty_counter = 0
object_present_latched = False
pending_detection_since = None

# Detection stabilization
last_good_detection = None
last_good_detection_miss_count = 0

# =========================
# Helpers
# =========================
def resize_frame(frame, target_width=640):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_fixed_roi(frame):
    h, w = frame.shape[:2]

    x1 = int(w * ROI_X1_RATIO)
    y1 = int(h * ROI_Y1_RATIO)
    x2 = int(w * ROI_X2_RATIO)
    y2 = int(h * ROI_Y2_RATIO)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise RuntimeError("ROI is empty. Check ROI values.")

    return roi


def sharpen_frame(frame):
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
    sharpened = cv2.addWeighted(frame, 1.6, blurred, -0.6, 0)
    return sharpened


def prepare_frame_for_vlm(frame):
    frame = frame.copy()
    h, w = frame.shape[:2]

    if w < MIN_VLM_WIDTH:
        scale = MIN_VLM_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    frame = sharpen_frame(frame)
    return frame


def encode_frame_to_base64(frame, jpeg_quality=55):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, buffer = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def extract_json_from_text(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Model did not return valid JSON.")


def normalize_material(material):
    material = str(material).strip().lower()

    aliases = {
        "other": "others",
        "unknown": "others",
        "misc": "others",
        "miscellaneous": "others",
        "others": "others",
        "undefined": "others",
    }

    if material in aliases:
        material = aliases[material]

    if material not in ALLOWED_MATERIALS:
        material = "others"

    return material


def is_ignored_object(obj_name):
    text = str(obj_name).strip().lower()

    if text == "none":
        return True

    for keyword in IGNORE_OBJECT_KEYWORDS:
        if keyword in text:
            return True

    return False


def validate_result(data):
    if not isinstance(data, dict):
        raise ValueError("Parsed result is not a JSON object.")

    if "object" not in data or "material" not in data:
        raise ValueError("Missing required keys: object/material")

    obj = str(data["object"]).strip().lower()
    material = normalize_material(data["material"])

    if obj in {"other", "others", "unknown", ""}:
        obj = "none"
        material = "others"

    return obj, material


def analyze_object(frame, client, max_retries=2):
    vlm_frame = prepare_frame_for_vlm(frame)
    image_b64 = encode_frame_to_base64(vlm_frame, jpeg_quality=JPEG_QUALITY)

    last_error = None

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                temperature=0,
                max_completion_tokens=80,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = extract_json_from_text(content)
            obj, material = validate_result(data)
            return obj, material, image_b64

        except Exception as e:
            last_error = e

    raise RuntimeError(f"ERROR: {last_error}")


def emit_detection_json(obj, material):
    output = {
        "object": obj,
        "material": material,
        "time": time.strftime("%H:%M:%S")
    }
    print(json.dumps(output, ensure_ascii=False))
    return output


def post_to_flask(material, obj, image_b64):
    payload = {
        "type": material,
        "object_name": obj,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_base64": image_b64
    }

    headers = {
        "x-api-key": API_KEY
    }

    response = requests.post(
        SERVER_URL,
        json=payload,
        headers=headers,
        timeout=10
    )

    print("POST status:", response.status_code, response.text)


def crop_detection_with_padding(frame, box, pad=12):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame.copy()
    return crop


def get_yolo_class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, str(class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detect_presence_with_yolo(roi_frame, yolo_model):
    roi_h, roi_w = roi_frame.shape[:2]
    roi_area = float(roi_h * roi_w)

    results = yolo_model.predict(
        source=roi_frame,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        verbose=False
    )

    if not results:
        return None

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_detection = None
    best_score = -1.0

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        class_name = get_yolo_class_name(yolo_model, cls_id).strip().lower()

        if class_name in YOLO_REJECT_CLASSES:
            continue

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        area_ratio = (box_w * box_h) / roi_area

        if area_ratio < YOLO_MIN_BOX_AREA_RATIO:
            continue

        score = area_ratio * conf

        if score > best_score:
            best_score = score
            best_detection = {
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "area_ratio": area_ratio
            }

    return best_detection


def stabilize_detection(current_detection):
    global last_good_detection
    global last_good_detection_miss_count

    if current_detection is not None:
        last_good_detection = current_detection.copy()
        last_good_detection_miss_count = 0
        return current_detection

    if last_good_detection is not None and last_good_detection_miss_count < DETECTION_HOLD_FRAMES:
        last_good_detection_miss_count += 1
        return last_good_detection

    last_good_detection = None
    last_good_detection_miss_count = 0
    return None


def can_send_now():
    return (time.time() - last_send_time) >= SEND_COOLDOWN_SECONDS


def mark_sent_now():
    global last_send_time
    last_send_time = time.time()


def draw_overlay(frame, detection=None, mode_text="EMPTY", present_count=0, empty_count=0, countdown=None):
    display = frame.copy()

    cv2.putText(
        display,
        f"State: {mode_text}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        f"PresentCount: {present_count}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        f"EmptyCount: {empty_count}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    if countdown is not None:
        cv2.putText(
            display,
            f"Delay: {countdown:.1f}s",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )

    if detection is not None:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Presence {detection['conf']:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return display

# =========================
# Threads
# =========================
def frame_reader(cap):
    global latest_frame

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.01)
            continue

        frame = resize_frame(frame, target_width=TARGET_WIDTH)

        with frame_lock:
            latest_frame = frame


def vlm_worker(client, object_crop):
    global latest_result, vlm_busy

    try:
        obj, material, image_b64 = analyze_object(object_crop, client)

        if is_ignored_object(obj):
            with result_lock:
                latest_result = None
            print("Ignored")
            return

        result = emit_detection_json(obj=obj, material=material)
        post_to_flask(material=material, obj=obj, image_b64=image_b64)

        with result_lock:
            latest_result = result

    except Exception as e:
        with result_lock:
            latest_result = None
        print(f"Error: {e}")

    finally:
        with vlm_lock:
            vlm_busy = False

# =========================
# Occupancy State Machine
# =========================
def process_presence_state(client, yolo_model, roi_frame):
    global present_counter
    global empty_counter
    global object_present_latched
    global pending_detection_since
    global vlm_busy

    raw_detection = detect_presence_with_yolo(roi_frame, yolo_model)
    detection = stabilize_detection(raw_detection)

    with vlm_lock:
        busy_now = vlm_busy

    countdown = None

    if not object_present_latched:
        if detection is not None:
            present_counter += 1
            empty_counter = 0

            if present_counter >= REQUIRED_PRESENT_FRAMES:
                if pending_detection_since is None:
                    pending_detection_since = time.time()
                    return detection, "DELAY_START", DETECTION_DELAY_SECONDS

                elapsed = time.time() - pending_detection_since
                countdown = max(0.0, DETECTION_DELAY_SECONDS - elapsed)

                if elapsed >= DETECTION_DELAY_SECONDS:
                    if not busy_now and can_send_now():
                        object_present_latched = True
                        mark_sent_now()
                        pending_detection_since = None

                        with vlm_lock:
                            vlm_busy = True

                        object_crop = crop_detection_with_padding(
                            roi_frame,
                            detection["box"],
                            pad=YOLO_CROP_PADDING
                        )

                        threading.Thread(
                            target=vlm_worker,
                            args=(client, object_crop),
                            daemon=True
                        ).start()

                        return detection, "SENT", 0.0

                    return detection, "WAIT_GROQ", 0.0

                return detection, "WAIT_5S", countdown

            pending_detection_since = None
            return detection, "CONFIRMING_PRESENT", None

        present_counter = 0
        empty_counter += 1
        pending_detection_since = None
        return None, "EMPTY", None

    if detection is None:
        empty_counter += 1
        present_counter = 0
        pending_detection_since = None

        if empty_counter >= REQUIRED_EMPTY_FRAMES:
            object_present_latched = False
            empty_counter = 0
            with result_lock:
                latest_result = None
            return None, "REARMED_EMPTY", None

        return None, "OCCUPIED_NO_YOLO", None

    empty_counter = 0
    present_counter = REQUIRED_PRESENT_FRAMES
    pending_detection_since = None
    return detection, "OCCUPIED", None

# =========================
# Main
# =========================
def main():
    global vlm_busy

    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_NEW_GROQ_API_KEY_HERE":
        raise RuntimeError("Please paste your Groq API key into GROQ_API_KEY.")

    print("Loading Groq client...")
    client = Groq(api_key=GROQ_API_KEY)

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream.")

    reader_thread = threading.Thread(target=frame_reader, args=(cap,), daemon=True)
    reader_thread.start()

    cv2.namedWindow("Tapo C211 Stream", cv2.WINDOW_NORMAL)

    print("Camera connected.")
    print("YOLO is used only for presence detection.")
    print("Groq runs only when ROI changes from empty to occupied.")
    print(f"Delay before Groq send: {DETECTION_DELAY_SECONDS:.1f} seconds")
    print("Press S to force manual scan.")
    print("Press Q to exit.")

    try:
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                blank = np.full((360, 480, 3), 255, dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Waiting for camera...",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 140, 255),
                    1,
                    cv2.LINE_AA
                )
                cv2.imshow("Tapo C211 Stream", blank)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            try:
                roi_frame = crop_fixed_roi(frame)
            except Exception as e:
                print(f"Error: ROI error: {e}")
                roi_frame = frame.copy()

            detection, mode_text, countdown = process_presence_state(client, yolo_model, roi_frame)

            display = draw_overlay(
                roi_frame,
                detection=detection,
                mode_text=mode_text,
                present_count=present_counter,
                empty_count=empty_counter,
                countdown=countdown
            )
            cv2.imshow("Tapo C211 Stream", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                with vlm_lock:
                    busy_now = vlm_busy

                if busy_now:
                    print("Busy")
                else:
                    with vlm_lock:
                        vlm_busy = True

                    if detection is not None:
                        object_crop = crop_detection_with_padding(
                            roi_frame,
                            detection["box"],
                            pad=YOLO_CROP_PADDING
                        )
                    else:
                        object_crop = roi_frame.copy()

                    threading.Thread(
                        target=vlm_worker,
                        args=(client, object_crop),
                        daemon=True
                    ).start()

            elif key == ord("q"):
                break

            if cv2.getWindowProperty("Tapo C211 Stream", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()