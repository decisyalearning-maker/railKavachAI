import os
from dotenv import load_dotenv
load_dotenv()

TARGETS = {
    "person","car","truck","bus","motorcycle","bicycle",
    "cow","horse","sheep","dog","elephant","bear"
}

_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(os.getenv("YOLO_MODEL","yolov8n.pt"))
    return _model

def detect_intrusions(frame):
    if os.getenv("ENABLE_VISION","true").lower() != "true":
        return frame, []

    model = get_model()
    conf = float(os.getenv("YOLO_CONFIDENCE","0.45"))
    result = model.predict(source=frame, conf=conf, verbose=False)[0]
    names = model.names
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label = names[cls_id]
        if label not in TARGETS:
            continue
        confidence = float(box.conf[0].item())
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "label": label,
            "confidence": round(confidence,3),
            "bbox": [x1,y1,x2,y2]
        })

    annotated = result.plot()
    return annotated, detections
