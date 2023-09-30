import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, modComplexity=1, detectionCon=0.5, trackCon=0.5):   
        self.mode = mode
        self.maxHands = maxHands
        self.modComplexity = modComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modComplexity, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                            self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                            self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        return img
    
    def findPosition(self, img, handNumber=0, draw=True):
        self.lmList = []
        bbox = []
        xList = []
        yList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    #if id == 4:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        c1, c2 = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (c1, c2), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, c1, c2]
    
    # def fingersUp(self):
    #     fingers = []
    #     # Thumb
    #     if self.lmList[4][1] > self.lmList[3][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
        
    #     # Fingers
    #     for id in range(1, 5):
    #         if self.lmList[4*id][2] < self.lmList[4*id-2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
        
    #     return fingers

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    #detector = handDetector()

    while True:
        ret, img = cap.read()
        #img = detector.findHands(img)
        #lmList = detector.findPosition(img, draw=True)
        #if len(lmList) != 0:
        #    print(lmList[4])
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img,f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()