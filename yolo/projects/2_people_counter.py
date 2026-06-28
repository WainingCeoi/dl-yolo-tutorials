"""Project 2: People Counter"""
import cv2
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter

# Config Parameters
model = YOLO("./yolo_models/yolo26m.pt")
cap = cv2.VideoCapture("./datasets/videos/people.mp4")

class_id = [0]
confidence = 0.5
device = "mps"
line_points = [[150, 500], [680, 320]]

# Initial Models
counter = ObjectCounter(
    model=model,
    classes=class_id,
    conf=confidence,
    region=line_points,
    device=device,
    show=True
)
counter.in_text = "Going Up"
counter.out_text = "Going Down"

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video frame is empty or Processing is complete.")
        break
    counter(img)

cap.release()
cv2.destroyAllWindows()
