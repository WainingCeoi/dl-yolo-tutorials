"""Project 1: Car Counter"""
import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter

# Load the pre-trained YOLO model
model = YOLO("../yolo_models/yolo26s.pt")
cap = cv2.VideoCapture("./datasets/videos/cars.mp4")

# Specific objects with string instead of theirs ID to enhance readability.
classes_name = list(model.names.values())
targeted_obj = ["car", "motorcycle", "bus", "truck"]
classes_idx = [classes_name.index(obj) for obj in targeted_obj] # Obtain object ID

confidence = 0.5
device = "cpu"
# Define Region of Interest (ROI)
region = [[430, 260], [700, 260], [670, 600], [20, 600]]

# Initial and configure car counter
counter = ObjectCounter(model=model, region=region, classes=classes_idx, conf=confidence,
                        device=device, show=True)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or Processing is complete.")
        break
    results = counter(frame)

cap.release()
cv2.destroyAllWindows()
