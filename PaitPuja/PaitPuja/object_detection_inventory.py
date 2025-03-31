import cv2
import numpy as np
import csv
import datetime
import os

# ----------------------------
# 1. Load YOLO model and classes
# ----------------------------
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out = net.getUnconnectedOutLayers()
if isinstance(unconnected_out[0], (list, np.ndarray)):
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out]

# ----------------------------
# 2. Define allowed food classes for logging
# ----------------------------
food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

# ----------------------------
# 3. Setup CSV file for inventory logging
# ----------------------------
csv_filename = "inventory_log.csv"
if not os.path.exists(csv_filename) or os.stat(csv_filename).st_size == 0:
    with open(csv_filename, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Timestamp", "Label", "Count"])

log_file = open(csv_filename, "a", newline="")
csv_writer = csv.writer(log_file)

# ----------------------------
# 4. Initialize video capture (using built-in camera at index 1)
# ----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# This dictionary will track food items that are already logged.
# Key: food label, Value: count (number detected) in the previous frame.
active_foods = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # ----------------------------
    # 5. Run YOLO object detection on the frame
    # ----------------------------
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ----------------------------
    # 6. Group allowed food detections by label for logging
    # ----------------------------
    allowed_counts = {}  # will hold counts for each allowed food item in this frame
    if len(indices) > 0:
        for i in indices.flatten():
            label = classes[class_ids[i]]
            # Always draw the bounding box on the frame, regardless of label.
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # For logging, count only if it's an allowed food item.
            if label in food_items:
                allowed_counts[label] = allowed_counts.get(label, 0) + 1

    # ----------------------------
    # 7. Log new food detections (debounced logging)
    # ----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # For each allowed food item in the current frame, log it only if it wasn't already logged.
    for label, count in allowed_counts.items():
        if label not in active_foods:
            csv_writer.writerow([timestamp, label, count])
            active_foods[label] = count
        else:
            # Optionally: If the count changes significantly, you could log an update here.
            pass

    # Remove items from active_foods that are no longer detected.
    for label in list(active_foods.keys()):
        if label not in allowed_counts:
            del active_foods[label]

    # ----------------------------
    # 8. Show the annotated frame
    # ----------------------------
    cv2.imshow("Object Detection (All Boxes Shown)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
