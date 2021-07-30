import cv2
import time
import numpy as np
import HandTracking_module as htm
import autopy

#####################################
_RED = (54, 67, 244)
_GREEN = (118, 230, 0)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (174, 164, 144)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)

wCam, hCam = 960, 520
frameR = 100  # frame reduction
pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

wScreen, hScreen = autopy.screen.size()
smoothening = 5
# print(wScreen, hScreen)
#####################################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector()

while True:
    # Find the landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # index finger and get 2 variables
        x2, y2 = lmList[12][1:]  # middle finger and get 2 variables

        # Check which fingers are up
        fingers = detector.fingerUps()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), _BLUE, 2)
        print(wCam- frameR, hCam -frameR)
        # Only finger index: moving mouse
        if fingers[1] == 1 and fingers[0] == 1:

            # Convert coordinate
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))
            # Smoothen values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening
            # move mouse
            autopy.mouse.move(cLocX, cLocY)
            cv2.circle(img, (x1, y1), 10, _PURPLE, cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY
        # CLick mode: both thumb and index fingers are up
        # Find distance between finger
        if fingers[1] == 1 and fingers[0] == 0:
            length, img, _ = detector.findDistance(4, 8, img)
            print(length)
            # Click mouse if distance short
            if length < 30:
                autopy.mouse.click()



    # Frame rate
    cTime = time.time()
    fbs = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fbs)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, _GREEN, 2)

    # Display
    cv2.imshow("Img", img)
    cv2.waitKey(2)
