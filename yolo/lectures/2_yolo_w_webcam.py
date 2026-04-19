"""Lecture 2: Yolo with Webcam"""
from ultralytics import YOLO


# For Webcam
video_source =0 # 0 for Mac, 1 for iPhone (Continuity Camera)

model = YOLO("../yolo_models/yolo26s.pt")

result = model.predict(source=0, show=True, save=False, device="cpu")
