"""Project 3: Personal Protective Equipment Detector"""
from ultralytics import YOLO


# For webcam
video_source = 0 # 0 for Mac, 1 for iPhone (Continuity Camera)

# For video input
# video_source = "../videos/ppe-3-1.mp4"

model = YOLO("../trained_models/ppe_detector/weights/best.pt")

result = model.predict(source=video_source, show=True, save=False, device="cpu")
