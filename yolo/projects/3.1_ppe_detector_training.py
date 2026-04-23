"""Project 3: Personal Protective Equipment Detector"""
from ultralytics import YOLO

# Custom data training
model = YOLO("./yolo_models/yolo26n.pt")

results = model.train(
    data="./datasets/construction_site_safety/data.yaml",
    epochs=5,
    name="ppe_detector",
    imgsz=640,
    workers = 6,
    device = "cpu",
    resume = True
)
