import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # 0 is internal camera, while 1 is external camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# initiate time
pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks) # testing any notification if there is any detection

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # each id have proper landmark
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # center point
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # draw connection between 2 dots

    # time for fbs
    cTime = time.time()
    fbs = 1/(cTime - pTime)
    pTime = cTime

    # put txt on the image
    cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_ITALIC, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
