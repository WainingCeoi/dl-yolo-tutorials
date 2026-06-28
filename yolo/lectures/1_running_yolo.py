"""Lecture 1: Running Yolo"""
from ultralytics import YOLO

model = YOLO("./yolo_models/yolo26m.pt")
results = model.predict("./datasets/images/1.jpg", save=False, device="mps")[0].show()
