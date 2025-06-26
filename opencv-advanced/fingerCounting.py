import cv2
import time
import os
import modules.handTrackingModule as HTM

#################
wCam,hCam = 640,480
#################

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    cap.set(3,wCam)
    cap.set(4,hCam)
    overlayList = []
    folderPath = "images"

    myList = os.listdir(folderPath)
    for imPath in myList:
        image = cv2.imread(f"{folderPath}/{imPath}")
        overlayList.append(image)
    
    tipIds = [4,8,12,16,20]

    # print(len(overlayList))
    
    detector = HTM.handDetector(detectionCon=0.7)
    while True:
        success, img = cap.read()

        if not success or img is None:
            print("Warning: Failed to grab frame from webcam.")
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False) 

        if len(lmList) != 0:
            fingers = []

            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            fingerCount = sum(fingers)

            if 0<= fingerCount <= len(overlayList):
                h,w,_ = overlayList[0].shape
                img[0:h,0:w] = overlayList[fingerCount-1]

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"{str(int(fps))} FPS", (1100, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()