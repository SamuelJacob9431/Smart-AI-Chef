import cv2

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam Feed", frame)  # Display the frame

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()