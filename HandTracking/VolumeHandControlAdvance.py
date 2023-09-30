import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Parameters
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    ret, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0 and lmList[0][2] > lmList[8][2]:
        #print(lmList[0], lmList[8])

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 220 < area < 1000:

            # Find distance between index and thumb
            length, img, info = detector.findDistance(4, 8, img)
            #print(length)

            # Convert Volume
            # Hand range 50 - 200
            # Volume Range -65 - 0
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            
            # Make smoother
            smoothness = 5
            volPer = smoothness * round(volPer/smoothness)

            # Check fingers up
            # fingers = detector.fingersUp()
            # print(fingers)

            # If pink is down set volume
            volume.SetMasterVolumeLevelScalar(volPer/100, None)

            if length < 50:
                cv2.circle(img, (info[4], info[5]), 10, (0, 0, 255), cv2.FILLED)
            if length > 200:
                cv2.circle(img, (info[4], info[5]), 10, (0, 255, 0), cv2.FILLED)
    
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow('frame', img)


    if cv2.waitKey(1) == ord('q'):
        break
