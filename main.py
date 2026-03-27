import cv2
import time
import csv
import os
from utils import get_direction, draw_text

os.makedirs("logs", exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

start_time = None
prev_direction = "Center"
alert_count = 0
current_status = "Monitoring..."  # shared state for webpage

log_file = open('logs/log.csv', 'a', newline='')
writer = csv.writer(log_file)


def get_status():
    return {"alert_count": alert_count, "status": current_status}


def generate_frames():
    global start_time, prev_direction, alert_count, current_status

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            face_cx = x + fw // 2
            face_cy = y + fh // 2
            frame_cx = w // 2
            frame_cy = h // 2

            direction = get_direction(face_cx, face_cy, frame_cx, frame_cy, fw, fh)

            if direction != prev_direction:
                prev_direction = direction
                start_time = None

            draw_text(frame, f"Looking: {direction}", (30, 40), (0, 255, 0))
            draw_text(frame, f"Alerts: {alert_count}", (30, 80), (255, 255, 0))

            if direction != "Center":
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > 2:  # 2 seconds
                    alert_count += 1
                    current_status = f"CHEATING DETECTED! Looking {direction}"
                    draw_text(frame, "CHEATING ALERT!", (30, 120), (0, 0, 255))
                    writer.writerow([time.ctime(), direction, "Cheating Alert"])
                    log_file.flush()
                    start_time = time.time()
            else:
                current_status = "Monitoring..."
                start_time = None

        else:
            draw_text(frame, "No Face Detected!", (30, 40), (0, 0, 255))
            current_status = "No Face Detected!"

            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 2:  # 2 seconds
                alert_count += 1
                writer.writerow([time.ctime(), "No Face", "Face Absent Alert"])
                log_file.flush()
                start_time = time.time()

            draw_text(frame, f"Alerts: {alert_count}", (30, 80), (255, 255, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')