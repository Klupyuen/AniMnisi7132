import cv2

CAMERA_IP = "192.168.0.2"
CAMERA_USERNAME = "Paratiro"
CAMERA_PASSWORD = "ParaTiro321"

RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:554/stream1"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open RTSP stream.")
    raise SystemExit

print("Camera connected successfully. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("RTSP Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()