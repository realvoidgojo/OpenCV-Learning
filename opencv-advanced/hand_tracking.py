import cv2
import time
import modules.handTrackingModule as HTM

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HTM.handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()