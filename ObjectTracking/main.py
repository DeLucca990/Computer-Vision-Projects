import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720, 500: 800]

    # Object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi, (x,y), (x+w, y+h), (0, 0, 255), 3)

            detections.append([x, y, w, h])

    # Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, h, w, id = box_id
        cv2.putText(roi, str(box_id[4]), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(30)

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()