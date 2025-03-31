# yolo_kitchen_detection.py

import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can also try yolov8s.pt for better accuracy

# Open webcam (change to 0 or video path if needed)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow('Kitchen Item Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
