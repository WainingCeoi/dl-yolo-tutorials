"""Lecture 2: Yolo with Webcam"""
from ultralytics import YOLO

# For Webcam
video_source =0 # 0 for Mac, 1 for iPhone (Continuity Camera)

model = YOLO("./models/pretrained/yolo26m.pt")

result = model.predict(source=video_source, show=True, save=False, device="mps")
