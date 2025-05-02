import cv2
import time
import winsound
from ultralytics import YOLO

model = YOLO("best.pt")  # Your custom trained model

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

out = cv2.VideoWriter("weapon_detection_output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# States
high_chances_start = 0
very_confident_start = 0
very_confident_count = 0
beeped = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes

    current_time = time.time()
    detected = False
    confidence_percent = 0

    if boxes and len(boxes.conf) > 0:
        confidence = float(boxes.conf[0])
        confidence_percent = int(confidence * 100)
        detected = True

        # Update status based on confidence
        if confidence_percent > 90:
            color = (0, 255, 0)  # Green
            very_confident_start = current_time
            very_confident_count += 1
            if not beeped:
                winsound.Beep(1000, 200)
                beeped = True
        elif confidence_percent > 80 and (current_time - very_confident_start) > 10:
            color = (0, 255, 255)  # Yellow
            high_chances_start = current_time
            beeped = False
        elif (current_time - very_confident_start) > 10:
            color = (0, 0, 255)  # Red
            beeped = False

        # Draw confidence percentage
        cv2.putText(
            annotated_frame,
            f"Gun: {confidence_percent}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # Blinking logic
    blink_on = int(current_time * 2) % 2 == 0

    # Display alerts with proper priorities and blinking
    if current_time - very_confident_start <= 10:
        if blink_on:
            msg = "ALERT: Gun Detected! (Very Confident)"
            if very_confident_count > 1:
                msg += f" [x{very_confident_count}]"
            cv2.putText(
                annotated_frame,
                msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
    elif current_time - high_chances_start <= 5:
        if blink_on:
            cv2.putText(
                annotated_frame,
                "ALERT: Gun Detected! (High Chances)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

    out.write(annotated_frame)
    cv2.imshow("Weapon Detection - Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
