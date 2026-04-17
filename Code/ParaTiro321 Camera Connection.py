import cv2

camera_ip = "172.20.10.5"
username = "Cylops"
password = "22Yu11meng"

rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/stream1"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("Tapo C211 Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()