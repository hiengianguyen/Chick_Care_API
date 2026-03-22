import os
import random
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from module_1 import FlockMonitor
from module_2 import BehaviorAnalyzer

import serial
import json

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
)

# ====== Kết nối Roboflow (dùng cho /api/analyze-frame) ======
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "9Jn7ADj8ghfbbPwGxKS2")
MODEL_ID = os.environ.get("CHICK_CARE_MODEL_ID", "chicken-zqstb/2")

_flock_monitor = None
_behavior_analyzer = None
_inference_client = None

# Trạng thái stream camera & kết quả mới nhất
_camera = None
_stream_thread = None
_stream_running = False
_latest_frame_jpeg = None  # bytes ảnh JPEG đã annotate
_latest_data = None        # dict kết quả phân tích mới nhất

def get_inference_client():
    global _inference_client
    if _inference_client is None:
        from inference_sdk import InferenceHTTPClient
        _inference_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
    return _inference_client


def get_flock_monitor():
    global _flock_monitor
    if _flock_monitor is None:
        _flock_monitor = FlockMonitor()
    return _flock_monitor


def get_behavior_analyzer():
    global _behavior_analyzer
    if _behavior_analyzer is None:
        _behavior_analyzer = BehaviorAnalyzer()
    return _behavior_analyzer


def _camera_loop():
    """
    Vòng lặp nền: đọc frame từ camera, chạy Roboflow + FlockMonitor + BehaviorAnalyzer,
    lưu kết quả vào biến global để UI chỉ cần poll.
    """
    global _camera, _stream_running, _latest_frame_jpeg, _latest_data

    if _camera is None:
        # 0: default webcam, có thể chỉnh index / đường dẫn RTSP tuỳ nhu cầu
        _camera = cv2.VideoCapture(0)

    _stream_running = True

    while _stream_running:
        ok, frame = _camera.read()
        if not ok:
            # Chờ một chút rồi đọc lại nếu lỗi tạm thời
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        try:
            client = get_inference_client()
            result = client.infer(frame, model_id=MODEL_ID)
        except Exception:
            # Nếu lỗi inference thì bỏ qua frame này
            time.sleep(0.1)
            continue

        predictions = result.get("predictions", [])

        # Vẽ bounding box lên frame (YOLO/Roboflow format: x,y = center, width, height)
        annotated = frame.copy()
        for pred in predictions:
            # Gán ID ngẫu nhiên 00–100 cho mỗi đối tượng (nếu chưa có)
            obj_id = pred.get("id")
            if obj_id is None:
                obj_id = random.randint(0, 100)
                pred["id"] = obj_id

            x_c = pred.get("x", 0)
            y_c = pred.get("y", 0)
            bw = pred.get("width", 0)
            bh = pred.get("height", 0)

            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            label = pred.get("class", "chicken")
            conf = pred.get("confidence", 0)
            color = (0, 255, 0)  # BGR xanh lá

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} #{int(obj_id):02d} {conf:.0%}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            time.sleep(0.05)
            continue

        monitor = get_flock_monitor()
        stable_count, missing_alert, crowding_alert = monitor.process(
            predictions, w, h
        )

        analyzer = get_behavior_analyzer()
        alerts = analyzer.analyze(predictions)

        _latest_frame_jpeg = buf.tobytes()
        _latest_data = {
            "predictions": [
                {
                    "id": p.get("id"),
                    "x": p.get("x"),
                    "y": p.get("y"),
                    "width": p.get("width"),
                    "height": p.get("height"),
                    "class": p.get("class"),
                    "confidence": p.get("confidence"),
                }
                for p in predictions
            ],
            "predictions_count": len(predictions),
            "stable_count": stable_count,
            "missing_alert": missing_alert,
            "crowding_alert": round(crowding_alert, 2),
            "alerts": alerts,
        }

        # Điều chỉnh sleep để kiểm soát FPS
        time.sleep(0.05)  # ~20 FPS


@app.post("/api/start-stream")
def start_stream():
    """
    Bật luồng camera + phân tích trên server.
    Gọi một lần khi UI mở.
    """
    global _stream_thread, _stream_running

    if _stream_thread is None or not _stream_thread.is_alive():
        _stream_running = True
        _stream_thread = threading.Thread(target=_camera_loop, daemon=True)
        _stream_thread.start()

    return jsonify({"status": "stream_started"})


@app.get("/api/stream-frame")
def stream_frame():
    """
    Trả về frame mới nhất dạng ảnh JPEG để UI hiển thị.
    """
    if _latest_frame_jpeg is None:
        return jsonify({"error": "No frame available yet"}), 503
    return Response(_latest_frame_jpeg, mimetype="image/jpeg")


@app.get("/api/latest-data")
def latest_data():
    """
    Trả về dữ liệu phân tích mới nhất (predictions, alerts, ...).
    """
    if _latest_data is None:
        return jsonify({"error": "No data available yet"}), 503
    return jsonify(_latest_data)

arduino = serial.Serial("COM3",9600)
@app.get("/api/get_data")
def data_ardruino():
    data = request.args.get("d")
    arduino.write((data + "\n").encode())
    
    return jsonify({
        "message": "Success" + " " +  data
    })

@app.get("/api/sensor")
def get_sensor():
    try:
        data = arduino.readline().decode().strip()
        parsed = json.loads(data)

        return jsonify(parsed)
    except:
        return jsonify({
            "error": "invalid data"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
