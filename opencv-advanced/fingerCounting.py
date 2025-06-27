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
    fingerCount = ""

    myList = os.listdir(folderPath)
    for imPath in myList:
        image = cv2.imread(f"{folderPath}/{imPath}")
        overlayList.append(image)
        print(f"{folderPath}/{imPath}")
    
    tipIds = [4,8,12,16,20]

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
            
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:  
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            fingerCount = fingers.count(1)

            if fingerCount >= 0 and fingerCount < len(overlayList):
                overlay_index = 5 if fingerCount == 0 else fingerCount - 1
                h,w,_ = overlayList[overlay_index].shape
                img[0:h,0:w] = overlayList[overlay_index]
                cv2.putText(img, str(fingerCount), (50, 300), cv2.FONT_HERSHEY_PLAIN, 5,
                            (255, 0, 255), 5)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"{str(int(fps))} FPS", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()