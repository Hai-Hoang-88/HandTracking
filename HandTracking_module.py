"""
"https://google.github.io/mediapipe/solutions/hands"
Module of Hand tracking
Hai Hoang
"""

import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks) # testing any notification if there is any detection

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # draw connection between 2 dots
        return img

    def findPosition(self, img, handNo=0, draw=True, slef=None):
        xList = []
        yList = []
        bbox = []  # boundary of object
        self.lmList = []
        if self.result.multi_hand_landmarks:
            # get the first hand
            myHand = self.result.multi_hand_landmarks[handNo]
            # each id have proper landmark
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # center point
                xList.append(cx)  # must append list other while errors
                yList.append(cy)  # must append list other while errors
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return self.lmList, bbox

    def fingerUps(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=7, t=2):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            # create circles at fingers and line connect
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # create circle at the center point of line

        length = math.hypot(x2 - x1, y2 - y1)  # measure distance between 2 points

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # initiate time
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 0 is internal camera, while 1 is external camera
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)  # get para img and put in definition
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # time for fbs
        cTime = time.time()
        fbs = 1 / (cTime - pTime)
        pTime = cTime

        # put txt on the image
        cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_ITALIC, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
