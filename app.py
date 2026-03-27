from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
import time
import csv
import os

app = Flask(__name__)
os.makedirs("logs", exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

state = {
    "alert_count": 0,
    "status": "Monitoring...",
    "start_time": None,
    "prev_direction": "Center"
}

log_file = open('logs/log.csv', 'a', newline='')
writer = csv.writer(log_file)


def get_direction(face_cx, face_cy, frame_cx, frame_cy, fw, fh):
    tx = fw * 0.25
    ty = fh * 0.25
    if face_cx < frame_cx - tx:
        return "Left"
    elif face_cx > frame_cx + tx:
        return "Right"
    elif face_cy < frame_cy - ty:
        return "Up"
    elif face_cy > frame_cy + ty:
        return "Down"
    return "Center"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')

@app.route('/status')
def status():
    return jsonify({
        "status": state["status"],
        "alert_count": state["alert_count"]
    })

@app.route('/process', methods=['POST'])
def process():
    data = request.json.get('frame', '')
    if not data:
        return jsonify({"status": "No Frame", "face": None, "direction": None})

    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "Invalid Frame", "face": None, "direction": None})

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )

    now = time.time()

    if len(faces) > 0:
        (x, y, fw, fh) = faces[0]
        face_cx = x + fw // 2
        face_cy = y + fh // 2
        direction = get_direction(face_cx, face_cy, w // 2, h // 2, fw, fh)

        if direction != state["prev_direction"]:
            state["prev_direction"] = direction
            state["start_time"] = None

        if direction != "Center":
            if state["start_time"] is None:
                state["start_time"] = now
            elif now - state["start_time"] > 2:
                state["alert_count"] += 1
                state["status"] = f"CHEATING DETECTED! Looking {direction}"
                writer.writerow([time.ctime(), direction, "Cheating Alert"])
                log_file.flush()
                state["start_time"] = now
        else:
            state["status"] = "Monitoring..."
            state["start_time"] = None

        return jsonify({
            "status": state["status"],
            "alert_count": state["alert_count"],
            "direction": direction,
            "face": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)}
        })

    else:
        state["status"] = "No Face Detected!"
        if state["start_time"] is None:
            state["start_time"] = now
        elif now - state["start_time"] > 2:
            state["alert_count"] += 1
            writer.writerow([time.ctime(), "No Face", "Face Absent"])
            log_file.flush()
            state["start_time"] = now

        return jsonify({
            "status": state["status"],
            "alert_count": state["alert_count"],
            "direction": None,
            "face": None
        })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
