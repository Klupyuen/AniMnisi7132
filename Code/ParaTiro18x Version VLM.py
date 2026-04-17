import cv2
import json
import base64
import time
import threading
import numpy as np
from groq import Groq

# =========================
# Configuration
# =========================
GROQ_API_KEY =
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

CAMERA_IP = "192.168.0.2"
CAMERA_USERNAME = "Paratiro"
CAMERA_PASSWORD = "ParaTiro321"

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/stream1"

# ROI
ROI_X1_RATIO = 0.12
ROI_Y1_RATIO = 0.50
ROI_X2_RATIO = 0.92
ROI_Y2_RATIO = 0.98

# Display / image prep
TARGET_WIDTH = 640
JPEG_QUALITY = 55
MIN_VLM_WIDTH = 512

# Presence detector settings
BACKGROUND_ALPHA = 0.02
DIFF_THRESHOLD = 25
MIN_CHANGED_AREA_RATIO = 0.008
MIN_CONTOUR_AREA_RATIO = 0.004
MORPH_KERNEL_SIZE = 5
OBJECT_CROP_PADDING = 12

# State machine
REQUIRED_PRESENT_FRAMES = 4
REQUIRED_EMPTY_FRAMES = 10
DETECTION_DELAY_SECONDS = 5.0
SEND_COOLDOWN_SECONDS = 2.0

ALLOWED_MATERIALS = {"paper", "glass", "metal", "plastic", "others"}

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

present_counter = 0
empty_counter = 0
object_present_latched = False
pending_detection_since = None

background_gray = None

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
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
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
            return obj, material

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

# =========================
# Occupancy Detector
# =========================
def preprocess_presence_frame(roi_frame):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def initialize_background_if_needed(gray):
    global background_gray
    if background_gray is None:
        background_gray = gray.astype(np.float32)


def update_background_model(gray):
    global background_gray
    if background_gray is None:
        background_gray = gray.astype(np.float32)
        return
    cv2.accumulateWeighted(gray, background_gray, BACKGROUND_ALPHA)


def detect_presence_with_background(roi_frame):
    global background_gray

    gray = preprocess_presence_frame(roi_frame)
    initialize_background_if_needed(gray)

    bg_uint8 = cv2.convertScaleAbs(background_gray)
    diff = cv2.absdiff(gray, bg_uint8)

    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    changed_ratio = float(np.count_nonzero(mask)) / mask.size

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, gray, mask, changed_ratio

    roi_h, roi_w = roi_frame.shape[:2]
    roi_area = float(roi_h * roi_w)

    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    largest_area_ratio = largest_area / roi_area

    if changed_ratio < MIN_CHANGED_AREA_RATIO and largest_area_ratio < MIN_CONTOUR_AREA_RATIO:
        return None, gray, mask, changed_ratio

    x, y, w, h = cv2.boundingRect(largest)

    detection = {
        "box": (x, y, x + w, y + h),
        "changed_ratio": changed_ratio,
        "area_ratio": largest_area_ratio
    }
    return detection, gray, mask, changed_ratio


def can_send_now():
    return (time.time() - last_send_time) >= SEND_COOLDOWN_SECONDS


def mark_sent_now():
    global last_send_time
    last_send_time = time.time()

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
        obj, material = analyze_object(object_crop, client)

        if is_ignored_object(obj):
            with result_lock:
                latest_result = None
            print("Ignored")
            return

        result = emit_detection_json(obj=obj, material=material)

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
# State Machine
# =========================
def process_presence_state(client, roi_frame):
    global present_counter
    global empty_counter
    global object_present_latched
    global pending_detection_since
    global vlm_busy

    detection, gray, mask, changed_ratio = detect_presence_with_background(roi_frame)

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
                    return detection, mask, changed_ratio, "DELAY_START", DETECTION_DELAY_SECONDS

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
                            pad=OBJECT_CROP_PADDING
                        )

                        threading.Thread(
                            target=vlm_worker,
                            args=(client, object_crop),
                            daemon=True
                        ).start()

                        return detection, mask, changed_ratio, "SENT", 0.0

                    return detection, mask, changed_ratio, "WAIT_GROQ", 0.0

                return detection, mask, changed_ratio, "WAIT_5S", countdown

            return detection, mask, changed_ratio, "CONFIRMING_PRESENT", None

        present_counter = 0
        empty_counter += 1
        pending_detection_since = None
        update_background_model(gray)
        return None, mask, changed_ratio, "EMPTY", None

    if detection is None:
        empty_counter += 1
        present_counter = 0
        pending_detection_since = None

        if empty_counter >= REQUIRED_EMPTY_FRAMES:
            object_present_latched = False
            empty_counter = 0
            with result_lock:
                latest_result = None
            update_background_model(gray)
            return None, mask, changed_ratio, "REARMED_EMPTY", None

        return None, mask, changed_ratio, "OCCUPIED_NO_OBJECT", None

    empty_counter = 0
    present_counter = REQUIRED_PRESENT_FRAMES
    pending_detection_since = None
    return detection, mask, changed_ratio, "OCCUPIED", None

# =========================
# Display
# =========================
def draw_overlay(frame, detection=None, mode_text="EMPTY", present_count=0, empty_count=0, countdown=None, changed_ratio=None):
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

    if changed_ratio is not None:
        cv2.putText(
            display,
            f"Changed: {changed_ratio:.4f}",
            (10, 84),
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
            (10, 106),
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
            f"Presence",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return display


def make_mask_bgr(mask):
    if mask is None:
        return None
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# =========================
# Main
# =========================
def main():
    global vlm_busy

    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
        raise RuntimeError("Please paste your Groq API key into GROQ_API_KEY.")

    print("Loading Groq client...")
    client = Groq(api_key=GROQ_API_KEY)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream.")

    reader_thread = threading.Thread(target=frame_reader, args=(cap,), daemon=True)
    reader_thread.start()

    cv2.namedWindow("Tapo C211 Stream", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Occupancy Mask", cv2.WINDOW_NORMAL)

    print("Camera connected.")
    print("Occupancy detector is ON.")
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

            detection, mask, changed_ratio, mode_text, countdown = process_presence_state(client, roi_frame)

            display = draw_overlay(
                roi_frame,
                detection=detection,
                mode_text=mode_text,
                present_count=present_counter,
                empty_count=empty_counter,
                countdown=countdown,
                changed_ratio=changed_ratio
            )
            cv2.imshow("Tapo C211 Stream", display)

            mask_display = make_mask_bgr(mask)
            if mask_display is not None:
                cv2.imshow("Occupancy Mask", mask_display)

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
                            pad=OBJECT_CROP_PADDING
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