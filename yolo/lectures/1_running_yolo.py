"""Lecture 1: Running Yolo"""
from ultralytics import YOLO


model = YOLO("../yolo_models/yolo11s.pt")
results = model("../images/1.jpg")[0].show()
