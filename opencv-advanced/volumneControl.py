import cv2
import time
import modules.handTrackingModule as HTM
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER


####################
wCAM , hCAM = 1280,720
####################

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Audio device found
# - Muted: False
# - Volume level: 0.0 dB
# - Volume range: -63.5 dB - 0.0 dB

# print("Audio device found")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
# volume.SetMasterVolumeLevel(-20.0, None)

volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0,None)

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3,wCAM)
    cap.set(4,hCAM)
    detector = HTM.handDetector(detectionCon=0.7)
    lmList = []

    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False)
        if len(lmList) != 0:
            # print(lmList[4],lmList[8])
            x1,y1 = lmList[4][1], lmList[4][2]
            x2,y2 = lmList[8][1], lmList[8][2]
            cx,cy = (x1 + x2)//2 , (y1+y2)//2
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,255,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3,cv2.FILLED)
            cv2.circle(img,(cx,cy),15,(0,255,255),cv2.FILLED)

            length = math.hypot(x2-x1,y2-y1)
            if length<50:
                cv2.circle(img,(cx,cy),15,(0,0,255),cv2.FILLED)
            elif length>300:
                cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
            # print(length)

            vol = np.interp(length,[50,300],[minVol,maxVol])
            volBar = np.interp(length,[50,300],[400,150])
            volPer = np.interp(length,[50,300],[0,100])

            # print(length,vol)
            volume.SetMasterVolumeLevel(vol,None)

        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{int(volPer)}%",(40,130),cv2.FONT_HERSHEY_PLAIN,2,
                    (255,0,0),3)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            volume.SetMasterVolumeLevel(0,None)
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()