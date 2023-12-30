import cv2
import numpy as np

cap = cv2.VideoCapture("escalator.mp4")
counter = 0
flag = False

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1100,720),)

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_th = cv2.adaptiveThreshold(frame_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((8, 8), np.uint8)
    frame_dil = cv2.dilate(frame_th, kernel, iterations=2)

    x, y, w, h = 490, 230, 30, 150
    roi = frame_dil[y:y+h, x:x+w]
    whites = cv2.countNonZero(roi)

    if whites > 4000 and flag == True:
        counter += 1
    if whites < 4000:
        flag = True
    else:
        flag = False

    
    if flag == False:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.rectangle(frame_th, (x, y), (x + w, y + h), (255, 255, 255), 6)

    cv2.putText(frame,str(whites),(x-30,y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.putText(frame, str(counter), (x + 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #print(counter)
    cv2.imshow("frame", frame)
    cv2.imshow("frame_dil", frame_th)
    cv2.imshow("roi", roi)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break