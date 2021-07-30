import cv2
import numpy as np
import time
import HandTracking_module as htm
import math

# import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##################################
# setup width and height of camera
wCam, hCam = 1200, 1200
##################################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.9)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()  # -65 and 0.03
minVol = volumeRange[0]  # assign the min to min volumeRange
maxVol = volumeRange[1]  # assign the max to max volumeRange
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # detect hand
    img = detector.findHands(img,)
    # detect position
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:  # make sure it will run only detect the hand
        # print(lmList[4], lmList[8])  # get the position of the thumb and index finger

        x1, y1 = lmList[4][1], lmList[4][2]  # x, y coordinate of thumb finger
        x2, y2 = lmList[8][1], lmList[8][2]  # x, y coordinate of index finger
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # create circles at fingers and line connect
        cv2.circle(img, (x1, y1), 7, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2, y2), (255,0,255), 2)
        cv2.circle(img, (cx,cy), 7, (255, 0, 255), cv2.FILLED)  # create circle at the center point of line

        length = math.hypot(x2-x1, y2-y1)  # measure distance between 2 points
        # print(length)

        # Hand range 50 - 300
        # volume range -65 - 0
        # map 2 ranges
        vol = np.interp(length, [5, 200], [minVol, maxVol])
        volBar = np.interp(length, [5, 200], [400, 150])
        volPer = np.interp(length, [5, 200], [0, 100])
        # print(vol)
        # set the range of volume
        # Need to define the condition only when the changes any volume is set
        volume.SetMasterVolumeLevel(vol, None)


        # pop up when 2 fingers touched
        if length < 50:
            cv2.circle(img, (cx,cy), 7, (0, 255, 0), cv2.FILLED)

    # Draw volume indicator
    cv2.rectangle(img, (50, 150), (85, 400), (210,210, 210), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (240, 128, 128), cv2.FILLED)
    cv2.putText(img, f':{int(volPer)} % ', (40, 450), cv2.FONT_HERSHEY_DUPLEX,
                1, (240, 128, 128), 2)

    # FBS text
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FBS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_DUPLEX,
                1, (255,0,255), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(45)
