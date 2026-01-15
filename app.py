from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)
model = YOLO("best.pt")

cap = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_video", methods=["POST"])
def set_video():
    global cap

    if cap:
        cap.release()

    file = request.files["video"]
    temp_path = "temp_video.mp4"
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)

    return ("", 204)

def generate_frames():
    global cap

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            break

        # 🔥 Resize
        frame = cv2.resize(frame, (400, 200))

        results = model(frame)
        frame = results[0].plot()

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
