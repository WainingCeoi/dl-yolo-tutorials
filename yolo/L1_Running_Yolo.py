"""Lecture 1: Running Yolo"""

from ultralytics import YOLO


model = YOLO("Yolo_Models/yolo11x.pt")
results = model("Images/1.jpg")
results[0].show()
