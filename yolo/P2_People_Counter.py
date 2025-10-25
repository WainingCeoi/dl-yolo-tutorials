"""Project 2: People Counter"""
import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter


model = YOLO("Yolo_Models/yolo11n.pt")
cap = cv2.VideoCapture("Videos/people.mp4")

class_id = [0]
confidence = 0.5
device = "mps"
line = [[150, 500], [680, 320]]

counter = ObjectCounter(model=model, classes=class_id, conf=confidence, region=line, device=device, show=True)

while True:
    success, img = cap.read()
    if not success:
        print("Video frame is empty or Processing is complete.")
        break
    counter(img)
    cv2.waitKey(0)