import cv2
import mediapipe as mp
import time


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
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    #if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

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