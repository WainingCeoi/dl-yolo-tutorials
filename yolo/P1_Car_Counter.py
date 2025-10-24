"""Project 1: Car Counter"""
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter
import cv2


# Load the pre-trained YOLO model
model = YOLO("Yolo_Models/yolo11s.pt")
cap = cv2.VideoCapture("Videos/cars.mp4")

# Specific objects with string instead of theirs ID to enhance readability.
classes_name = list(model.names.values())
targeted_obj = ["car", "motorcycle", "bus", "truck"]
classes_idx = [classes_name.index(obj) for obj in targeted_obj] # Obtain object ID

confidence = 0.5
device = "mps"
# Define Region of Interest (ROI)
region = [[430, 260], [700, 260], [670, 600], [20, 600]]

# Initial and configure car counter
counter = ObjectCounter(model=model, region=region, classes=classes_idx, conf=confidence,
                        device=device, show=True)


while True:
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or Processing is complete.")
        break
    results = counter(frame)
