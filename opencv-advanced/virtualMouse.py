import cv2
import numpy as np
import pyautogui
import time
from modules import handTrackingModule as HTM

wCam, hCam = 1280, 720
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HTM.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

def main():
    global plocX, plocY, pTime
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
