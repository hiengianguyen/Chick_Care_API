import config
from flask_socketio import SocketIO
import os
import random
import time
import threading

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

import requests

from module_1 import FlockMonitor
from module_2 import BehaviorAnalyzer

import time
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore

import cloudinary
import cloudinary.uploader

from flask import send_file
from gtts import gTTS
import io


cloudinary.config(
    cloud_name=config.CLOUD_NAME,
    api_key=config.API_KEY_CLOUDINARY,
    api_secret=config.API_SECRET_CLOUDINARY
)

ESP32_IP = config.ESP32_IP  # IP ESP32

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*",async_mode="threading")

CORS(
    app,
    resources={r"/api/*": {"origins": ["*"]}},
)

# ====== Kết nối  FIRESTORE ======
cred = credentials.Certificate("Database/key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# ====== Kết nối Roboflow (dùng cho /api/analyze-frame) ======
ROBOFLOW_API_KEY = config.ROBOFLOW_API_KEY
MODEL_ID = config.MODEL_ID

_flock_monitor = None
_behavior_analyzer = None
_inference_client = None

# Trạng thái stream camera & kết quả mới nhất
_camera = None
_stream_thread = None
_stream_running = False
_latest_frame_jpeg = None  # bytes ảnh JPEG đã annotate
_latest_data = None        # dict kết quả phân tích mới nhất
_latest_sensor_data = {}   # dict dữ liệu sensor từ Arduino

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

def upload_to_cloudinary(frame):
    _, buffer = cv2.imencode('.jpg', frame)

    result = cloudinary.uploader.upload(
        buffer.tobytes(),
        folder="chicken_alerts"
    )
    
    return result["secure_url"]

last_save_time = 0
def _camera_loop():
    # Vòng lặp nền: đọc frame từ camera, chạy Roboflow + FlockMonitor + BehaviorAnalyzer,
    # lưu kết quả vào biến global để UI chỉ cần poll.
    global _camera, _stream_running, _latest_frame_jpeg, _latest_data

    if _camera is None:
        # 0: default webcam, có thể chỉnh index / đường dẫn RTSP tuỳ nhu cầu
        _camera = cv2.VideoCapture(1)

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
        stationaryLength = 0
        separationLength = 0

        # Nếu có alerts, upload ảnh và lưu notification
        if alerts:
            global last_save_time
            now = time.time()
            if now - last_save_time < 60:
                continue
            last_save_time = now

            try:
                image_url = upload_to_cloudinary(annotated)
                for alert in alerts:
                    # Tìm obj_id gần nhất với vị trí alert
                    alert_x, alert_y = alert.get('x', 0), alert.get('y', 0)
                    nearest_obj_id = None
                    min_distance = float('inf')
                    for pred in predictions:
                        pred_x, pred_y = pred.get('x', 0), pred.get('y', 0)
                        distance = ((pred_x - alert_x) ** 2 + (pred_y - alert_y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            nearest_obj_id = int(pred.get('id', 0))
                    
                    obj_id_str = f"#{nearest_obj_id}" if nearest_obj_id is not None else "#unknown"
                    
                    if alert['type'] == 'separation':
                        separationLength += 1
                        title = "PHÁT HIỆN GÀ CÓ DẤU HIỆU BẤT THƯỜNG"
                        shortTitle = "Gà tách đàn"
                        message = f"Hệ thống phát hiện gà {obj_id_str} di chuyển tách khỏi đàn, có thể do yếu, bệnh hoặc bị ảnh hưởng bởi môi trường. Người dùng nên kiểm tra và theo dõi các cá thể này để đảm bảo an toàn cho toàn bộ đàn."
                    elif alert['type'] == 'stationary':
                        stationaryLength += 1
                        title = "PHÁT HIỆN GÀ CÓ DẤU HIỆU BẤT THƯỜNG"
                        shortTitle = "Gà đứng im"
                        message = f"Hệ thống ghi nhận con gà {obj_id_str} có dấu hiện đứng im trong một khoảng thời gian dài. Người dùng nên kiểm tra trực tiếp để xác định nguyên nhân và có biện pháp xử lý kịp thời."
                    else:
                        title = "PHÁT HIỆN GÀ CÓ DẤU HIỆU BẤT THƯỜNG"
                        shortTitle = "Gà có dấu hiệu bất thường"
                        message = "Hệ thống phát hiện có đấu hiệu bất thường vui lòng kiểm tra"
                    
                    notification_data = {
                        "title": title,
                        "shortTitle": shortTitle,
                        "message": message,
                        "imageUrl": image_url,
                        "isDeleted": False,
                        "isRead": False,
                        "createdAt": datetime.utcnow().isoformat() + 'Z',
                        "updatedAt": datetime.utcnow().isoformat() + 'Z',
                    }
                    write_time, doc_ref= db.collection("notiAlerts").add(notification_data)
                    notification_data["id"] = doc_ref.id
                    socketio.emit("chicken_alert", notification_data)
            except Exception as e:
                print(f"Error uploading to Cloudinary or saving notification: {e}")

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
            "stationaryLength": stationaryLength,
            "separationLength": separationLength,
        }

        # Điều chỉnh sleep để kiểm soát FPS
        time.sleep(0.05)  # ~20 FPS


@app.post("/api/start-stream")
def start_stream():
    # Bật luồng camera + phân tích trên server. Gọi một lần khi UI mở.
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

last_states = {}
feed_activation_times = {}  # Theo dõi thời gian kích hoạt feed: {"feed": timestamp}

def check_and_send(config):
    global last_states
    state = {}

    device = config["device"]

    if last_states:
        for deviceLast, configLast in last_states.items():
            if deviceLast == device and configLast.get('isVoiceControl') == True:
                return False
                

    # Tính should_active dựa trên thời gian
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_day = ["CN","T2","T3","T4","T5","T6","T7"][now.weekday()+1 if now.weekday()<6 else 0]
    should_active = (
        current_time in config["times"] and
        current_day in config["selectedDays"]
    )

    if(config["active"] == False):
        state = {
            "device": device,
            "active": should_active,
            "power": config.get("power")
        }
    else: 
        state = {
            "device": device,
            "active": True,
            "power": config.get("power")
        }

    if device == "feed":
        amount = config.get("amount", "Vừa")
        duration_map = {"Ít": 5, "Vừa": 10, "Nhiều": 15}
        state["duration"] = duration_map.get(amount, 10)
        # Không gửi power, chỉ gửi duration
        if "power" in state:
            del state["power"]

        if config.get("mode") == "Thủ công":
            state["active"] = config["active"]
        else: 
            state["active"] = should_active

        # Xử lý duration: nếu active=true, ghi nhận thời gian; nếu quá lâu, tự động set active=false
        if state["active"] == True:
            # Ghi nhận thời gian kích hoạt nếu chưa có
            if "feed" not in feed_activation_times:
                feed_activation_times["feed"] = datetime.now()
        else:
            # Xóa thời gian kích hoạt khi tắt
            if "feed" in feed_activation_times:
                del feed_activation_times["feed"]

        # Kiểm tra nếu feed đã hoạt động quá lâu so với duration
        if "feed" in feed_activation_times:
            elapsed = (datetime.now() - feed_activation_times["feed"]).total_seconds()
            if elapsed >= state["duration"]:
                state["active"] = False
                # Đồng bộ lại global data để lần sau không dùng active thủ công
                if "feed" in data:
                    data["feed"]["active"] = False
                    # Nếu là chế độ hẹn giờ, xóa thời gian đã chạy khỏi mảng
                    data["feed"]["times"] = [t for t in data["feed"]["times"] if t != current_time]
                del feed_activation_times["feed"]

    if device == "fan":
        if config.get("mode") == "Thủ công":
            state["active"] = config["active"]
        elif config.get("mode") == "Tự động":
            temp = _latest_sensor_data.get("temperature", 25)
            if temp > int(config.get("thresholdTemp", 30)):
                state["active"] = True
            else:
                state["active"] = should_active
        else:  # Hẹn giờ
            state["active"] = should_active

    if device == "light":
        if config.get("mode") == "Thủ công":
            state["active"] = config["active"]
        elif config.get("mode") == "Tự động":
            temp = _latest_sensor_data.get("temperature", 25)
            if temp < int(config.get("thresholdTemp", 25)):
                state["active"] = True
            else:
                state["active"] = should_active
        else:  # Hẹn giờ
            state["active"] = should_active

    # SO SÁNH RIÊNG TỪNG THIẾT BỊ
    if (last_states.get(device) != state) or config.get('isEditTime'):
        send_to_esp32_device(state)
        print("SEND:", state)
        last_states[device] = state
        return True  

    return False

@app.get("/api/temp-sensor")
def get_sensor():
    """Trả về dữ liệu sensor mới nhất từ Arduino."""
    if _latest_sensor_data:
        return jsonify(_latest_sensor_data)
    else:
        return jsonify({"error": "No sensor data available"}), 503
    
@app.post("/api/sensor")
def sensor():
    global _latest_sensor_data
    data = request.json
    _latest_sensor_data = data
    return jsonify({"status": "ok"})
    
data = {}
@app.post("/api/get_data_device")
def light_sensor():
    global data
    data = request.json  # nhận object từ React
    
    # Lưu vào Firestore collection "devices" (update nếu tồn tại, tạo mới nếu không)
    try:
        for device_name, config in data.items():
            doc_ref = db.collection("devices").document(device_name)
            doc_ref.set(config, merge=True)
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return jsonify({"error": "Failed to save configuration"}), 500
    
    return jsonify({
        "data": data,
        "message": "Configuration saved successfully"
    })


def background_loop():
    while True:
        if data:
            for device, config in data.items():
                config["device"] = device  # gắn tên thiết bị
                if last_states.get(device) and last_states.get(device, {}).get('isVoiceControl') == True: # kiểm tra xem có đang trong lệnh nói
                    config['active'] = False
                if check_and_send(config): 
                    config["isEditTime"] = False
                    break

        time.sleep(1)

@app.delete("/api/noti/<string:noti_id>")
def delete_noti(noti_id):
    # Xóa vĩnh viễn thông báo khỏi Firestore.
    db.collection("notiAlerts").document(noti_id).delete()
    return jsonify({"id": noti_id, "deleted": True})

@app.post("/api/noti/read/<string:noti_id>")
def noti_readed_noti(noti_id):
    doc_ref = db.collection("notiAlerts").document(noti_id)
    doc_ref.update({
        "isRead": True,
    })
    return jsonify({"id": noti_id, "isRead": True})



@app.get("/api/noti-alerts")
def get_noti_alerts():
    # Lấy danh sách thông báo từ collection notiAlerts. Query param: limit (tùy chọn).
    limit = request.args.get("limit")
    query = db.collection("notiAlerts").order_by("createdAt", direction=firestore.Query.DESCENDING)
    
    if limit:
        try:
            limit_int = int(limit)
            query = query.limit(limit_int)
        except ValueError:
            return jsonify({"error": "Invalid limit parameter, must be an integer"}), 400
    
    docs = query.stream()
    alerts = []
    isRead = 0
    for doc in docs:
        item = doc.to_dict() or {}
        item["id"] = doc.id
        if item.get("isRead", True) == False:
            isRead += 1
        alerts.append(item)

    return jsonify({"isReadCount": isRead,"alerts": alerts})

@app.get("/api/devices")
def get_devices():
    """Lấy tất cả cấu hình devices từ collection 'devices'."""
    try:
        docs = db.collection("devices").stream()
        devices = {}
        for doc in docs:
            devices[doc.id] = doc.to_dict()
        return jsonify(devices)
    except Exception as e:
        print(f"Error fetching devices: {e}")
        return jsonify({"error": "Failed to fetch devices"}), 500

def send_to_esp32_device(data):
    try:
        res = requests.post(
            f"{ESP32_IP}/control",
            json=data,
            timeout=2
        )
        print("ESP32:", res.text)
    except Exception as e:
        print("Loi gui ESP32:", e)

def handle_message(msg):
    global last_states
    msg = msg.lower()

    if "bật quạt" in msg:
        dataConfig = {"device": "fan", "active": True, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = True
        last_states['fan'] = dataConfig
        data.get('fan',{})['active'] = True
        return {"action": "fan_on", "reply": "Đã bật quạt"}
    elif "tắt quạt" in msg:
        dataConfig = {"device": "fan", "active": False, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = False
        last_states['fan'] = dataConfig
        data.get('fan',{})['active'] = False
        return {"action": "fan_off", "reply": "Đã tắt quạt"}
    elif "bật đèn" in msg:
        dataConfig = {"device": "light", "active": True, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = True
        last_states['light'] = dataConfig
        data.get('light',{})['active'] = True
        return {"action": "light", "reply": "Đã bật đèn sửi ấm"}
    elif "tắt đèn" in msg:
        dataConfig = {"device": "light", "active": False, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = False
        last_states['light'] = dataConfig
        data.get('light',{})['active'] = False
        return {"action": "light", "reply": "Đã tắt đèn sửi ấm"}
    elif "dừng cho ăn" in msg:
        dataConfig = {"device": "feed", "active": False, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = False
        last_states['feed'] = dataConfig
        data.get('feed',{})['active'] = False
        return {"action": "feed", "reply": "Đã tắt chế độ cho ăn"}
    elif "cho ăn" in msg:
        dataConfig = {"device": "feed", "active": True, "power": 100}
        send_to_esp32_device(dataConfig)
        dataConfig['isVoiceControl'] = True
        last_states['feed'] = dataConfig
        data.get('feed',{})['active'] = True
        return {"action": "feed", "reply": "Đã bật chế độ cho ăn"}
    elif "thông tin" in msg:
        return {"action": "tempHum", "reply": "Hiện tại hệ thống ghi nhận, số gà được phát hiện trong môi trường là "
                + ((str(_latest_data.get("predictions_count", "0")) + ' con') if _latest_data else '0 con') 
                + ", mật độ cao nhất là " 
                + ((str(_latest_data.get("crowding_alert", "0")) + '%') if _latest_data else '0%') 
                + ", số gà đứng yên là "
                + ((str(_latest_data.get("stationaryLength", "0")) + ' con') if _latest_data else '0 con')
                + ", số gà tách đàn là "
                + ((str(_latest_data.get("separationLength", "0")) + ' con') if _latest_data else '0 con') 
                +", nhiệt độ là "+ str(_latest_sensor_data.get("temperature", 0))+ " độ C" +" và độ ẩm là " + str(_latest_sensor_data.get("humidity", 0)) + '%'}
    else:
        return {"action": "unknown", "reply": "Tôi không hiểu lệnh"}
 

# 🌐 API chatbot
@app.post("/api/chat")
def chat():
    msg = request.json["message"]

    result = handle_message(msg)

    return jsonify({
        "reply": result["reply"]
    })

@app.get("/api/tts")
def tts():
    text = request.args.get("text")

    mp3_fp = io.BytesIO()
    tts = gTTS(text, lang='vi')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    return send_file(mp3_fp, mimetype="audio/mpeg")

# Bắt đầu background thread cho background_loop
threading.Thread(target=background_loop, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
