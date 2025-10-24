"""Lecture 2: Yolo with Webcam"""

from ultralytics import YOLO
import cv2


# For Webcam
cap = cv2.VideoCapture(0) # 0 for Mac, 1 for iPhone (Continuity Camera)
cap.set(3, 1280)
cap.set(4, 720)

# For Video
# cap = cv2.VideoCapture("Videos/bikes.mp4")

model = YOLO("Yolo_Models/yolo11s.pt")

while True:
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or Processing is complete.")
        break
    results = model(frame, stream=True, device="mps")
    for result in results:
        cv2.imshow("YOLO Detection Model (Press \"Q\" to Quit)", result.plot())
    if cv2.waitKey(1) == ord('q'):
        break
