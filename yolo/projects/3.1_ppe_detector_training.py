"""Project 3: Personal Protective Equipment Detector"""
from ultralytics import YOLO

# Custom data training
model = YOLO("./yolo_models/yolo26m.pt")

results = model.train(
    data="./datasets/construction_site_safety/data.yaml",
    epochs=50,
    name="ppe_detector",
    imgsz=640,
    workers = 6,
    device = "mps",
    resume = True
)
