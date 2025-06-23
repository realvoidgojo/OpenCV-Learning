import cv2
import time
import scripts.modules.poseTrackingModule as PM

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success,img = cap.read()
        if not success:
            break

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.QT_FONT_BOLD,2,(0,255,255),3)

    cv2.imshow("img",img)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()

