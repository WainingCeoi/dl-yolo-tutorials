"""Project 3: Personal Protective Equipment Detector"""
from ultralytics import YOLO


# Custom data training
model = YOLO("../yolo_models/yolo26s.pt")

results = model.train(
    data="../datasets/construction_site_safety/data.yaml",
    epochs=50,
    imgsz=640,
    batch=10,
    workers = 4,
    device = "mps"
)
