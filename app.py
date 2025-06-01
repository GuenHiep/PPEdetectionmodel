from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import base64
import cloudinary
import cloudinary.uploader
import cloudinary.api


# Khởi tạo Firebase
cred = credentials.Certificate("finalprj-92f33-firebase-adminsdk-fbsvc-d6a638b37a.json")  # 🔑 Đặt file JSON này trong cùng thư mục
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://finalprj-92f33-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app = Flask(__name__)
CORS(app)

# Tải mô hình YOLO
model = YOLO("best.pt")

# Class name mapping
SELECTED_CLASSES = {
    0: "Earmuffs",
    1: "Face",
    2: "Face mask",
    3: "Face-guard",
    4: "Foot",
    5: "Glasses",
    6: "Gloves",
    7: "Hands",
    8: "Head",
    9: "Helmet",
    10: "Medical-suit",
    11: "Person",
    12: "Safety vest",
    13: "Safety-suit",
    14: "Tools"
}

REQUIRED_PPE = {"Helmet", "Safety vest", "Gloves"}

cloudinary.config( 
    cloud_name = "dhydhjie2", 
    api_key = "936998357839651", 
    api_secret = "Z7ZMI_rILP7dThBwR0YaCETilsQ",
    secure=True
)

def upload_image_to_cloudinary(image_path):
    try:
        response = cloudinary.uploader.upload(image_path)
        print("Upload thành công.")
        return response['secure_url']
    except Exception as e:
        print("Upload thất bại:", e)
        return None


# Hàm push dữ liệu vi phạm lên Firebase
# def push_to_firebase(missing_ppe, image_path):
#     try:
#         with open(image_path, "rb") as img_file:
#             image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         data = {
#             "timestamp": timestamp,
#             "missing_ppe": list(missing_ppe),
#             "image_base64": image_base64
#         }

#         ref = db.reference("violations")
#         ref.push(data)
#         print("Cảnh báo đã được đẩy lên Firebase.")
#     except Exception as e:
#         print("Lỗi khi đẩy dữ liệu lên Firebase:", e)

def push_to_firebase(missing_ppe, image_path):
    image_url = upload_image_to_cloudinary(image_path)
    if image_url:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "timestamp": timestamp,
            "missing_ppe": list(missing_ppe),
            "image_url": image_url
        }
        ref = db.reference("violations")
        ref.push(data)
        print("Đẩy dữ liệu lên Firebase thành công.")
    else:
        print("Không thể upload ảnh, dữ liệu không được đẩy lên Firebase.")


@app.route('/detect', methods=['POST'])
def detect_ppe():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image_file = request.files['image']
    image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(image_np, conf=0.25, iou=0.35)[0]

    detected_classes = set()
    detections = []

    for box in results.boxes.data:
        x1, y1, x2, y2, score, class_id = map(float, box)
        class_id = int(class_id)
        if class_id in SELECTED_CLASSES:
            class_name = SELECTED_CLASSES[class_id]
            detected_classes.add(class_name)
            detections.append({
                "class_name": class_name,
                "score": score,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_np, class_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = "output.jpg"
    cv2.imwrite(output_path, image_np)

    missing_ppe = REQUIRED_PPE - detected_classes
    if missing_ppe:
        push_to_firebase(missing_ppe, output_path)

    return jsonify({
        "detections": detections,
        "missing_ppe": list(missing_ppe),
        "image_url": "/output.jpg"
    })

@app.route('/output.jpg')
def get_output_image():
    return send_file("output.jpg", mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
