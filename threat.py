# import cv2
# import time
# import threading
# import winsound
# import numpy as np
# from imutils.video import VideoStream
# from imutils.video import FPS
# from ultralytics import YOLO
# import imutils

# # Load MobileNetSSD for person detection
# print("[INFO] loading MobileNetSSD model...")
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor"]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# net = cv2.dnn.readNetFromCaffe(
#     'MobileNetSSD_deploy.prototxt.txt',
#     'MobileNetSSD_deploy.caffemodel'
# )

# # Load YOLOv5 for gun detection
# print("[INFO] loading YOLOv5 model...")
# model = YOLO('best.pt')  # Your gun detection model

# # Video stream
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# fps = FPS().start()

# # States
# high_chances_start = 0
# very_confident_start = 0
# very_confident_count = 0
# beep_thread_running = False
# last_beep_time = 0

# def beep_continuous():
#     global beep_thread_running, very_confident_start
#     while time.time() - very_confident_start <= 10:
#         winsound.Beep(1000, 200)
#         time.sleep(1)
#     beep_thread_running = False

# while True:
#     frame = vs.read()
#     frame = imutils.resize(frame, width=800)
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
#                                  0.007843, (300, 300), 127.5)
    
#     net.setInput(blob)
#     detections = net.forward()

#     # Detect persons
#     for i in np.arange(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.2:
#             idx = int(detections[0, 0, i, 1])
#             if CLASSES[idx] != "person":
#                 continue
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#             cv2.putText(frame, label, (startX, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

#     # Detect guns
#     results = model(frame)
#     boxes = results[0].boxes

#     detected = False
#     confidence_percent = 0

#     if boxes and len(boxes.conf) > 0:
#         confidence = float(boxes.conf[0])
#         confidence_percent = int(confidence * 100)
#         detected = True

#         current_time = time.time()

#         if confidence_percent > 90:
#             very_confident_start = current_time
#             very_confident_count += 1
#             if not beep_thread_running:
#                 beep_thread_running = True
#                 threading.Thread(target=beep_continuous, daemon=True).start()
#         elif confidence_percent > 80 and (current_time - very_confident_start) > 10:
#             high_chances_start = current_time

#         # Draw confidence percentage on screen
#         color = (0, 255, 0) if confidence_percent > 90 else (0, 255, 255) if confidence_percent > 80 else (0, 0, 255)
#         cv2.putText(
#             frame,
#             f"Gun: {confidence_percent}%",
#             (10, 110),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             color,
#             2
#         )

#     # Handle blinking alert
#     current_time = time.time()
#     blink_on = int(current_time * 2) % 2 == 0

#     if current_time - very_confident_start <= 10:
#         if blink_on:
#             alert_text = f"ALERT: Gun Detected! (Very Confident)"
#             if very_confident_count > 1:
#                 alert_text += f" [x{very_confident_count}]"
#             cv2.putText(
#                 frame,
#                 alert_text,
#                 (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 3
#             )
#     elif current_time - high_chances_start <= 5:
#         if blink_on:
#             cv2.putText(
#                 frame,
#                 "ALERT: Gun Detected! (High Chances)",
#                 (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 255),
#                 3
#             )

#     # Show frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break

#     fps.update()

# # Cleanup
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cv2.destroyAllWindows()
# vs.stop()












import cv2
import time
import threading
import winsound
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
from ultralytics import YOLO
import imutils

# Load MobileNetSSD for person detection
print("[INFO] loading MobileNetSSD model...")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt',
    'MobileNetSSD_deploy.caffemodel'
)

# Load YOLOv5 for gun detection
print("[INFO] loading YOLO model...")
model = YOLO('best.pt')  # Your trained gun detection model

# Video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# States for alarm
high_chances_start = 0
very_confident_start = 0
very_confident_count = 0
beep_thread_running = False

def beep_continuous():
    global beep_thread_running, very_confident_start
    while time.time() - very_confident_start <= 10:
        winsound.Beep(1000, 200)
        time.sleep(1)
    beep_thread_running = False

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]

    # Prepare blob and forward pass for person detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    persons = []  # List to hold person bounding boxes
    guns = []     # List to hold gun bounding boxes

    # Detect persons
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            persons.append((startX, startY, endX, endY))

    # Detect guns
    results = model(frame)
    boxes = results[0].boxes

    detected = False
    confidence_percent = 0

    if boxes and len(boxes.xyxy) > 0:
        for gun_box in boxes.xyxy:
            x1, y1, x2, y2 = gun_box[:4]
            guns.append((int(x1), int(y1), int(x2), int(y2)))

        # Take first gun box confidence
        confidence = float(boxes.conf[0])
        confidence_percent = int(confidence * 100)
        detected = True

        current_time = time.time()

        if confidence_percent > 90:
            very_confident_start = current_time
            very_confident_count += 1
            if not beep_thread_running:
                beep_thread_running = True
                threading.Thread(target=beep_continuous, daemon=True).start()
        elif confidence_percent > 80 and (current_time - very_confident_start) > 10:
            high_chances_start = current_time

    # Analyze distance between persons and guns
    for (startX, startY, endX, endY) in persons:
        person_center_x = (startX + endX) / 2
        person_center_y = (startY + endY) / 2
        threat = False

        for (gx1, gy1, gx2, gy2) in guns:
            gun_center_x = (gx1 + gx2) / 2
            gun_center_y = (gy1 + gy2) / 2

            # Euclidean distance between gun and person
            distance = np.sqrt((person_center_x - gun_center_x) ** 2 + (person_center_y - gun_center_y) ** 2)

            if distance < 150:  # Adjust threshold based on your scenario
                threat = True
                break

        # Draw bounding box
        if threat:
            color = (0, 0, 255)  # Red for threat
            label = "THREAT"
        else:
            color = (0, 255, 0)  # Green for normal person
            label = "Person"

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw gun bounding boxes
    for (gx1, gy1, gx2, gy2) in guns:
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        cv2.putText(frame, "Gun", (gx1, gy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display alert messages based on confidence
    current_time = time.time()
    blink_on = int(current_time * 2) % 2 == 0

    if current_time - very_confident_start <= 10:
        if blink_on:
            alert_text = f"ALERT: Gun Detected! (Very Confident)"
            if very_confident_count > 1:
                alert_text += f" [x{very_confident_count}]"
            cv2.putText(
                frame,
                alert_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
    elif current_time - high_chances_start <= 5:
        if blink_on:
            cv2.putText(
                frame,
                "ALERT: Gun Detected! (High Chances)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

    # Show confidence score if detected
    if detected:
        color = (0, 255, 0) if confidence_percent > 90 else (0, 255, 255) if confidence_percent > 80 else (0, 0, 255)
        cv2.putText(
            frame,
            f"Gun Confidence: {confidence_percent}%",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # Show the frame
    cv2.imshow("Weapon Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

# Cleanup
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
