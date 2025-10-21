from ultralytics import YOLO
import cv2


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("Yolo_Models/yolo11s.pt")

while True:
    success, frame = cap.read()
    results = model(frame, device="mps", stream=True)
    for result in results:
        cv2.imshow("Image", result.plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()