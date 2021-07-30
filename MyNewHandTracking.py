import cv2
import mediapipe as mp
import time
import HandTracking_module as htm

# initiate time
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # 0 is internal camera, while 1 is external camera
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)  # get para img and put in definition
    lmList = detector.findPosition(img, draw=False)  # if no draw=True, will draw a circle
    if len(lmList) != 0:
        print(lmList[4])

    # time for fbs
    cTime = time.time()
    fbs = 1/(cTime - pTime)
    pTime = cTime

    # put txt on the image
    cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_ITALIC, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)