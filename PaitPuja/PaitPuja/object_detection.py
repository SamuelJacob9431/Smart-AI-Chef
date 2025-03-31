import cv2
import numpy as np

# --- Load YOLO model and classes ---
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out = net.getUnconnectedOutLayers()
if isinstance(unconnected_out[0], (list, np.ndarray)):
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out]

# --- Camera setup (using built-in camera, index 1) ---
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# --- Smoothing parameters ---
alpha = 0.7  # weight for the current frame (0 < alpha < 1)
# We'll store detections from the previous frame as a list of dictionaries
prev_detections = []  # Each element: {'label': str, 'smoothed': [x, y, w, h]}

def get_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # --- Prepare blob and forward pass through YOLO ---
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # --- Gather detections ---
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # adjust threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # --- (Optional) Apply non-max suppression ---
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Prepare current detections with smoothing
    current_detections = []  # Each element: {'label': str, 'smoothed': [x, y, w, h]}
    if len(indices) > 0:
        for i in indices.flatten():
            label = classes[class_ids[i]]
            box = boxes[i]
            current_center = get_center(box)

            # Try to find a matching detection from previous frame (same label and nearby center)
            matched_box = None
            for prev in prev_detections:
                if prev['label'] == label:
                    prev_center = get_center(prev['smoothed'])
                    distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                    if distance < 50:  # threshold in pixels (adjust as needed)
                        matched_box = prev['smoothed']
                        break

            if matched_box is not None:
                # Smooth the coordinates with exponential moving average
                smoothed_box = [alpha * box[j] + (1 - alpha) * matched_box[j] for j in range(4)]
            else:
                smoothed_box = box

            current_detections.append({'label': label, 'smoothed': smoothed_box})

    # Update previous detections for the next frame
    prev_detections = current_detections

    # --- Draw the (smoothed) detections ---
    for detection in current_detections:
        label = detection['label']
        x, y, w, h = detection['smoothed']
        # Convert coordinates to integers for drawing
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection (Smoothed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
